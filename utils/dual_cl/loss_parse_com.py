import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label as scipy_label
from utils.dual_branch.simple_loss_parse import SparseSliceLoss


# =========================================================================
# 🫁 肺部专版：基于 2D 面积的形态感知对比学习 (Pulmonary Area-Adaptive CL)
# =========================================================================
class PulmonaryAreaAdaptivePatchContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feat_dim = int(cfg.get("feat_dim", 32))
        self.temperature = float(cfg.get("temperature", 0.1))
        dyn_temp_cfg = cfg.get("dynamic_temperature", {})
        self.dynamic_temperature = bool(dyn_temp_cfg.get("enable", False))
        self.min_temperature = float(dyn_temp_cfg.get("min_temperature", self.temperature))
        self.max_temperature = float(dyn_temp_cfg.get("max_temperature", self.temperature))
        if self.min_temperature > self.max_temperature:
            self.min_temperature, self.max_temperature = self.max_temperature, self.min_temperature
        self.num_patches = int(cfg.get("num_patches", 16))
        self.ignore_index = int(cfg.get("ignore_index", 255))

        self.roi_size = int(cfg.get("roi_size", cfg.get("patch_size", 16)))
        self.base_patch_size = int(cfg.get("base_patch_size", self.roi_size))
        self.search_size = int(cfg.get("search_size", 0))
        self.margin_ratio = float(cfg.get("margin_ratio", 0.25))
        self.min_margin = int(cfg.get("min_margin", 3))
        self.max_margin = int(cfg.get("max_margin", 16))
        self.min_component_pixels = int(cfg.get("min_component_pixels", 2))
        self.max_component_pixels = int(cfg.get("max_component_pixels", 100000))
        self.max_bbox_area_ratio = float(cfg.get("max_bbox_area_ratio", 1.0))
        self.min_resized_fg_pixels = int(cfg.get("min_resized_fg_pixels", 2))
        self.max_flood_iters = int(cfg.get("max_flood_iters", 256))
        self.near_bg_kernel = int(cfg.get("near_bg_kernel", 9))
        self.proto_momentum = float(cfg.get("proto_momentum", 0.95))

        # 🌟 肺部专有参数：面积判定阈值
        self.area_threshold = int(cfg.get("area_threshold", 400))

        self.register_buffer("vessel_proto", F.normalize(torch.randn(self.feat_dim), dim=0))
        self.register_buffer("bg_proto", F.normalize(torch.randn(self.feat_dim), dim=0))

    def _dynamic_temp_from_margin(self, pos_sim, neg_sim):
        if not self.dynamic_temperature:
            return torch.full_like(pos_sim, self.temperature)

        confidence = (pos_sim.detach() - neg_sim.detach()).abs().div(2.0).clamp(0.0, 1.0)
        return self.max_temperature - confidence * (self.max_temperature - self.min_temperature)

    def _binary_nce(self, pos_sim, neg_sim):
        temp = self._dynamic_temp_from_margin(pos_sim, neg_sim).clamp_min(1e-6)
        pos = torch.exp(pos_sim / temp)
        neg = torch.exp(neg_sim / temp)
        return -torch.log(pos / (pos + neg + 1e-8)).mean()

    def _resize_mask_to_feat(self, mask, feat):
        if mask.shape[2:] == feat.shape[2:]:
            return mask
        return F.interpolate(mask.float(), size=feat.shape[2:], mode="nearest")

    def _dynamic_trunk_separation_2d(self, mask2d_tensor):
        """
        [极速核心] 仅对选定的单张 2D 切片进行面积连通域判定，耗时 < 0.5ms。
        """
        device = mask2d_tensor.device
        m_cpu = mask2d_tensor[0].cpu().numpy()

        labeled_array, num_features = scipy_label(m_cpu == 1)
        new_m_cpu = m_cpu.copy()

        for i in range(1, num_features + 1):
            comp_mask = (labeled_array == i)
            if comp_mask.sum() > self.area_threshold:
                new_m_cpu[comp_mask] = 2  # 超过阈值的转为主干 2

        return torch.tensor(new_m_cpu, device=device).unsqueeze(0)

    @torch.no_grad()
    def update_prototypes(self, v_feats, b_feats):
        if v_feats is not None and v_feats.numel() > 0:
            v_mean = F.normalize(v_feats.mean(dim=0), dim=0)
            new_v = F.normalize(self.proto_momentum * self.vessel_proto + (1.0 - self.proto_momentum) * v_mean, dim=0)
            self.vessel_proto.copy_(new_v)
        if b_feats is not None and b_feats.numel() > 0:
            b_mean = F.normalize(b_feats.mean(dim=0), dim=0)
            new_b = F.normalize(self.proto_momentum * self.bg_proto + (1.0 - self.proto_momentum) * b_mean, dim=0)
            self.bg_proto.copy_(new_b)

    def _resize_roi(self, feat_patch, mask_patch):
        feat_patch = F.interpolate(feat_patch.unsqueeze(0), size=(self.roi_size, self.roi_size), mode="bilinear",
                                   align_corners=False).squeeze(0)
        mask_patch = F.interpolate(mask_patch.float().unsqueeze(0), size=(self.roi_size, self.roi_size),
                                   mode="nearest").squeeze(0)
        return feat_patch, mask_patch

    def _take_slice(self, feat3d, mask3d, direction, d, h, w):
        if direction == 0: return feat3d[:, d, :, :], mask3d[:, d, :, :], h, w
        if direction == 1: return feat3d[:, :, h, :], mask3d[:, :, h, :], d, w
        return feat3d[:, :, :, w], mask3d[:, :, :, w], d, h

    def _direction_from_valid_density(self, mask3d, coords):
        valid = (mask3d[0] == 1) | (mask3d[0] == 0)
        density_d, density_h, density_w = valid.sum(dim=(1, 2)), valid.sum(dim=(0, 2)), valid.sum(dim=(0, 1))
        return torch.argmax(
            torch.stack([density_d[coords[:, 0]], density_h[coords[:, 1]], density_w[coords[:, 2]]], dim=1), dim=1)

    def _flood_component_2d(self, fg_local, sy, sx):
        if fg_local.numel() == 0 or not bool(fg_local[sy, sx]): return None
        comp = torch.zeros_like(fg_local, dtype=torch.bool)
        comp[sy, sx] = True
        fg4 = fg_local[None, None].float()
        for _ in range(self.max_flood_iters):
            prev = comp
            grown = F.max_pool2d(comp[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0] > 0
            comp = grown & fg_local
            if torch.equal(comp, prev): break
        return comp & (fg4[0, 0] > 0)

    def _component_crop(self, feat2d, mask2d, ay, ax):
        _, H, W = feat2d.shape
        y0, y1, x0, x1 = 0, H, 0, W
        if self.search_size > 0:
            y0, y1 = max(0, int(ay) - self.search_size // 2), min(H, int(ay) + self.search_size // 2 + 1)
            x0, x1 = max(0, int(ax) - self.search_size // 2), min(W, int(ax) + self.search_size // 2 + 1)

        fg_local = mask2d[0, y0:y1, x0:x1] == 1
        comp_local = self._flood_component_2d(fg_local, int(ay) - y0, int(ax) - x0)

        if comp_local is None or not (
                self.min_component_pixels <= int(comp_local.sum().item()) <= self.max_component_pixels):
            return self._fixed_vessel_crop(feat2d, mask2d, ay, ax)

        coords = torch.nonzero(comp_local, as_tuple=False)
        yy0, yy1 = y0 + int(coords[:, 0].min()), y0 + int(coords[:, 0].max()) + 1
        xx0, xx1 = x0 + int(coords[:, 1].min()), x0 + int(coords[:, 1].max()) + 1

        margin = max(self.min_margin, min(int(round(max(yy1 - yy0, xx1 - xx0) * self.margin_ratio)), self.max_margin))
        yy0, yy1 = max(0, yy0 - margin), min(H, yy1 + margin)
        xx0, xx1 = max(0, xx0 - margin), min(W, xx1 + margin)

        if ((yy1 - yy0) * (xx1 - xx0)) / max(1, H * W) > self.max_bbox_area_ratio:
            return self._fixed_vessel_crop(feat2d, mask2d, ay, ax)

        feat_patch = feat2d[:, yy0:yy1, xx0:xx1]
        patch_mask = mask2d[:, yy0:yy1, xx0:xx1].clone()

        comp_crop = torch.zeros((H, W), dtype=torch.bool, device=mask2d.device)
        comp_crop[y0:y1, x0:x1] = comp_local
        comp_crop = comp_crop[yy0:yy1, xx0:xx1]

        other_vessels = ((patch_mask[0] == 1) | (patch_mask[0] == 2)) & (~comp_crop)
        patch_mask[0, other_vessels] = self.ignore_index
        patch_mask[0, comp_crop] = 1

        feat_patch, patch_mask = self._resize_roi(feat_patch, patch_mask)
        if int((patch_mask[0] == 1).sum().item()) < self.min_resized_fg_pixels:
            return self._fixed_vessel_crop(feat2d, mask2d, ay, ax)
        return feat_patch, patch_mask

    def _fixed_vessel_crop(self, feat2d, mask2d, ay, ax):
        _, H, W = feat2d.shape
        size = min(self.base_patch_size, H, W)
        y0, x0 = max(0, min(int(round(ay - size / 2)), H - size)), max(0, min(int(round(ax - size / 2)), W - size))
        feat_patch, mask_patch = feat2d[:, y0:y0 + size, x0:x0 + size], mask2d[:, y0:y0 + size, x0:x0 + size].clone()
        mask_patch[0, mask_patch[0] == 2] = self.ignore_index
        mask_patch[0, mask_patch[0] == 1] = 1
        return self._resize_roi(feat_patch, mask_patch)

    def _fixed_bg_crop(self, feat2d, mask2d, ay, ax):
        _, H, W = feat2d.shape
        size = min(self.base_patch_size, H, W)
        y0, x0 = max(0, min(int(round(ay - size / 2)), H - size)), max(0, min(int(round(ax - size / 2)), W - size))
        feat_patch, mask_patch = feat2d[:, y0:y0 + size, x0:x0 + size], mask2d[:, y0:y0 + size, x0:x0 + size].clone()
        mask_patch[0, (mask_patch[0] == 1) | (mask_patch[0] == 2)] = self.ignore_index
        return self._resize_roi(feat_patch, mask_patch)

    def sample_and_crop(self, feat, mask, is_gt=True):
        mask = self._resize_mask_to_feat(mask, feat)
        B, C, D, H, W = feat.shape
        v_patches_feat, v_patches_mask, b_patches_feat, b_patches_mask = [], [], [], []

        for b in range(B):
            f_b, m_b = feat[b], mask[b].clone()

            if is_gt:
                # =====================================================================
                # 🎯 有标签分支 (GT)：由于数据是极度稀疏的十字切片，绝大部分平面全是 255
                # 必须沿用旧逻辑：找点 -> 判断点周围密度 -> 沿着密度最大的方向切片
                # =====================================================================
                v_coords = torch.nonzero(m_b[0] == 1, as_tuple=False)
                b_coords = torch.nonzero(m_b[0] == 0, as_tuple=False)

                if v_coords.shape[0] > self.num_patches:
                    v_coords = v_coords[torch.randperm(v_coords.shape[0], device=v_coords.device)[:self.num_patches]]
                if b_coords.shape[0] > min(5000, max(self.num_patches, 1)):
                    b_coords = b_coords[torch.randperm(b_coords.shape[0], device=b_coords.device)[:5000]]
                if b_coords.shape[0] > self.num_patches:
                    b_coords = b_coords[torch.randperm(b_coords.shape[0], device=b_coords.device)[:self.num_patches]]

                v_dirs = self._direction_from_valid_density(m_b, v_coords) if v_coords.numel() > 0 else []
                b_dirs = self._direction_from_valid_density(m_b, b_coords) if b_coords.numel() > 0 else []

                for coord, direction in zip(v_coords.cpu().tolist(), v_dirs.cpu().tolist() if len(v_dirs) else []):
                    d, h, w = coord
                    feat2d, mask2d, ay, ax = self._take_slice(f_b, m_b, direction, d, h, w)
                    c_f, c_m = self._component_crop(feat2d, mask2d, int(ay), int(ax))
                    v_patches_feat.append(c_f.unsqueeze(0))
                    v_patches_mask.append(c_m.unsqueeze(0))

                for coord, direction in zip(b_coords.cpu().tolist(), b_dirs.cpu().tolist() if len(b_dirs) else []):
                    d, h, w = coord
                    feat2d, mask2d, ay, ax = self._take_slice(f_b, m_b, direction, d, h, w)
                    c_f, c_m = self._fixed_bg_crop(feat2d, mask2d, int(ay), int(ax))
                    b_patches_feat.append(c_f.unsqueeze(0))
                    b_patches_mask.append(c_m.unsqueeze(0))

            else:
                # =====================================================================
                # 🚀 伪标签分支 (Unlabeled)：稠密预测数据
                # 全新逻辑：随机抽平面 -> 平面划分 1 和 2 -> 在平面上直接采样 1 号点
                # =====================================================================
                v_collected = 0
                b_collected = 0
                max_retries = 20  # 防止没有血管的死循环

                for _ in range(max_retries):
                    if v_collected >= self.num_patches and b_collected >= self.num_patches:
                        break

                    # 1. 随机选取一个切面维度和索引
                    axis = torch.randint(0, 3, (1,)).item()
                    if axis == 0:
                        idx = torch.randint(0, D, (1,)).item()
                        feat2d = f_b[:, idx, :, :]
                        mask2d = m_b[:, idx, :, :]
                    elif axis == 1:
                        idx = torch.randint(0, H, (1,)).item()
                        feat2d = f_b[:, :, idx, :]
                        mask2d = m_b[:, :, idx, :]
                    else:
                        idx = torch.randint(0, W, (1,)).item()
                        feat2d = f_b[:, :, :, idx]
                        mask2d = m_b[:, :, :, idx]

                    # 2. 对这一整张 2D 切片执行极速面积划分 (区分出细支1 和 主干2)
                    mask2d_sep = self._dynamic_trunk_separation_2d(mask2d)

                    # 3. 找出当前切片上的 细小分支(1) 和 背景(0)
                    v_coords_2d = torch.nonzero(mask2d_sep[0] == 1, as_tuple=False)
                    b_coords_2d = torch.nonzero(mask2d_sep[0] == 0, as_tuple=False)

                    # 如果这个面上全是大块主干或者全是背景，直接重新抽面！
                    if v_coords_2d.shape[0] == 0:
                        continue

                    # 4. 打乱坐标，确保随机采样
                    v_coords_2d = v_coords_2d[torch.randperm(v_coords_2d.shape[0], device=v_coords_2d.device)]
                    b_coords_2d = b_coords_2d[torch.randperm(b_coords_2d.shape[0], device=b_coords_2d.device)]

                    # --- 直接从筛选好的点里切框，不再有排雷逻辑 ---
                    for ay, ax in v_coords_2d.cpu().tolist():
                        if v_collected >= self.num_patches: break
                        c_f, c_m = self._component_crop(feat2d, mask2d_sep, int(ay), int(ax))
                        v_patches_feat.append(c_f.unsqueeze(0))
                        v_patches_mask.append(c_m.unsqueeze(0))
                        v_collected += 1

                    for ay, ax in b_coords_2d.cpu().tolist():
                        if b_collected >= self.num_patches: break
                        c_f, c_m = self._fixed_bg_crop(feat2d, mask2d_sep, int(ay), int(ax))
                        b_patches_feat.append(c_f.unsqueeze(0))
                        b_patches_mask.append(c_m.unsqueeze(0))
                        b_collected += 1

        return v_patches_feat, v_patches_mask, b_patches_feat, b_patches_mask

    def _class_macro_feats(self, feats_list, masks_list, cls_value):
        if not feats_list: return None
        feats, masks = torch.cat(feats_list, dim=0), torch.cat(masks_list, dim=0)
        n, c = feats.shape[:2]
        cls_mask = (masks.reshape(n, -1) == cls_value).float()
        counts = cls_mask.sum(dim=1, keepdim=True)
        keep = counts.squeeze(1) > 0
        if not keep.any(): return None
        return torch.bmm(feats.reshape(n, c, -1)[keep], cls_mask[keep].unsqueeze(2)).squeeze(2) / (counts[keep] + 1e-6)

    def _near_background_mask(self, v_mask_flat, b_mask_flat, side):
        kernel = max(3, self.near_bg_kernel)
        if kernel % 2 == 0: kernel += 1
        near = F.max_pool2d(v_mask_flat.reshape(-1, 1, side, side).float(), kernel_size=kernel, stride=1,
                            padding=kernel // 2).reshape(-1, side * side) > 0
        return b_mask_flat.bool() & near

    def compute_macro_micro_loss(self, v_feats_list, v_masks_list, b_feats_list, b_masks_list):
        macro_loss = torch.tensor(0.0, device=self.vessel_proto.device)
        micro_loss = torch.tensor(0.0, device=self.vessel_proto.device)
        valid_macro, valid_micro = 0, 0

        # 仅针对 1(细支) 和 0(背景) 更新，完全无视 2
        v_macro = self._class_macro_feats(v_feats_list, v_masks_list, cls_value=1)
        if v_macro is not None:
            v_norm = F.normalize(v_macro, dim=1)
            pos_sim = torch.matmul(v_norm, self.vessel_proto)
            neg_sim = torch.matmul(v_norm, self.bg_proto)
            macro_loss = macro_loss + self._binary_nce(pos_sim, neg_sim)
            valid_macro += 1

        b_macro = self._class_macro_feats(b_feats_list, b_masks_list, cls_value=0)
        if b_macro is not None:
            b_norm = F.normalize(b_macro, dim=1)
            pos_sim = torch.matmul(b_norm, self.bg_proto)
            neg_sim = torch.matmul(b_norm, self.vessel_proto)
            macro_loss = macro_loss + self._binary_nce(pos_sim, neg_sim)
            valid_macro += 1

        if v_feats_list:
            ff, mm = torch.cat(v_feats_list, dim=0), torch.cat(v_masks_list, dim=0)
            n, c, h, w = ff.shape
            ff, mm = ff.reshape(n, c, -1), mm.reshape(n, -1)
            v_mask, b_mask = (mm == 1).float(), (mm == 0).float()
            keep = (v_mask.sum(dim=1) > 0) & (b_mask.sum(dim=1) > 0)

            if keep.any():
                f_keep, v_keep, b_keep = ff[keep], v_mask[keep], b_mask[keep]
                anchor = F.normalize(
                    torch.bmm(f_keep, v_keep.unsqueeze(2)).squeeze(2) / (v_keep.sum(dim=1, keepdim=True) + 1e-6), dim=1)
                pos_sim = torch.matmul(anchor, self.vessel_proto)

                near_bg = self._near_background_mask(v_keep, b_keep, h).float()
                fallback = near_bg.sum(dim=1) <= 0
                if fallback.any(): near_bg[fallback] = b_keep[fallback]

                f_norm = F.normalize(f_keep, dim=1)
                sim = torch.bmm(anchor.detach().unsqueeze(1), f_norm).squeeze(1)
                masked_sim = sim.masked_fill(near_bg <= 0, -1.0)
                hard_neg_sim = masked_sim.max(dim=1).values
                temp = self._dynamic_temp_from_margin(pos_sim, hard_neg_sim).clamp_min(1e-6)
                pos = torch.exp(pos_sim / temp)
                exp_sim = torch.exp(sim / temp.unsqueeze(1))
                neg = (exp_sim * near_bg).sum(dim=1) / (near_bg.sum(dim=1) + 1e-8)
                micro_loss = micro_loss + (-torch.log(pos / (pos + neg + 1e-8)).mean())
                valid_micro += 1

        return (macro_loss / max(1, valid_macro)) + (micro_loss / max(1, valid_micro))

    def forward(self, feat, mask, is_gt=True, update_proto=False):
        v_f, v_m, b_f, b_m = self.sample_and_crop(feat, mask, is_gt=is_gt)
        if update_proto: self.update_prototypes(self._class_macro_feats(v_f, v_m, cls_value=1),
                                                self._class_macro_feats(b_f, b_m, cls_value=0))
        return self.compute_macro_micro_loss(v_f, v_m, b_f, b_m)


class DualBranchLoss(nn.Module):
    def __init__(self, base_sup_loss, pseudo_loss_fn, cl_cfg=None, ramp_epochs=50, max_pseudo_weight=0.5,
                 pseudo_label_mode="hard"):
        super().__init__()
        self.sup_loss_fn = base_sup_loss
        self.pseudo_loss_fn = pseudo_loss_fn
        self.ramp_epochs = ramp_epochs
        self.max_pseudo_weight = max_pseudo_weight
        self.pseudo_label_mode = str(pseudo_label_mode).lower()
        self.cl_cfg = cl_cfg
        self.enable_cl = cl_cfg.get("enable", False) if cl_cfg else False
        cfg_for_pseudo = cl_cfg or {}
        self.pseudo_confidence = float(cfg_for_pseudo.get("pseudo_confidence", 0.75))
        self.use_reliable_agreement = bool(cfg_for_pseudo.get("use_reliable_agreement", True))
        self.max_branch_diff = float(cfg_for_pseudo.get("max_branch_diff", 0.2))
        self.min_fg_prob = float(cfg_for_pseudo.get("min_fg_prob", self.pseudo_confidence))
        self.max_bg_prob = float(cfg_for_pseudo.get("max_bg_prob", 1.0 - self.pseudo_confidence))

        if self.enable_cl:
            self.cl_loss_fn = PulmonaryAreaAdaptivePatchContrastiveLoss(cl_cfg)
            self.cl_weight = float(cl_cfg.get("weight", 0.2))
            self.warmup = int(cl_cfg.get("warmup_epochs", 20))

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length <= 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def _make_pseudo_and_reliable_masks(self, pred1_u, pred2_u):
        prob1 = torch.sigmoid(pred1_u)
        prob2 = torch.sigmoid(pred2_u)
        avg_prob = 0.5 * (prob1 + prob2)
        diff = torch.abs(prob1 - prob2)

        pseudo = (avg_prob > 0.5).float()
        fg_reliable = avg_prob >= self.min_fg_prob
        bg_reliable = avg_prob <= self.max_bg_prob
        reliable = fg_reliable | bg_reliable

        if self.use_reliable_agreement:
            agreement = (prob1 > 0.5) == (prob2 > 0.5)
            reliable = reliable & agreement

        if self.max_branch_diff is not None and self.max_branch_diff >= 0:
            reliable = reliable & (diff <= self.max_branch_diff)

        pseudo_mask = pseudo.clone()
        pseudo_mask[~reliable] = 255
        return pseudo, pseudo_mask

    def _pseudo_loss_targets(self, pred1_u, pred2_u):
        prob1 = torch.sigmoid(pred1_u)
        prob2 = torch.sigmoid(pred2_u)
        pseudo_1 = (prob1 > 0.5).float()
        pseudo_2 = (prob2 > 0.5).float()

        if self.pseudo_label_mode == "soft":
            alpha = torch.rand(
                size=(prob1.shape[0], 1, 1, 1, 1),
                device=prob1.device,
                dtype=prob1.dtype,
            )
            pseudo_mix = alpha * prob1 + (1.0 - alpha) * prob2
            loss_target_1 = pseudo_mix
            loss_target_2 = pseudo_mix
        else:
            loss_target_1 = pseudo_2
            loss_target_2 = pseudo_1

        return loss_target_1, loss_target_2

    def forward(self, preds_l, mask_l, preds_u, feats_l, feats_u, current_epoch, img_l=None):
        pred1_l, pred2_l = preds_l
        pred1_u, pred2_u = preds_u

        # 1. 监督损失
        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            loss_sup = 0.5 * (self.sup_loss_fn(pred1_l, mask_l, img_l) + self.sup_loss_fn(pred2_l, mask_l, img_l))
        else:
            loss_sup = 0.5 * (self.sup_loss_fn(pred1_l, mask_l) + self.sup_loss_fn(pred2_l, mask_l))

        # 2. 伪标签损失 (无视大块血管，因为计算伪标签时直接二值化)
        with torch.no_grad():
            loss_target_1, loss_target_2 = self._pseudo_loss_targets(pred1_u, pred2_u)
            _, reliable_pseudo = self._make_pseudo_and_reliable_masks(pred1_u, pred2_u)

        loss_ps = 0.5 * (self.pseudo_loss_fn(pred1_u, loss_target_1) + self.pseudo_loss_fn(pred2_u, loss_target_2))
        loss_ps = loss_ps * (self.max_pseudo_weight * self.sigmoid_rampup(current_epoch, self.ramp_epochs))
        # 2. 混合 Soft 伪标签损失
        # #    loss_ps 使用 pseudo_mix，不再使用 0.5 阈值截断后的 hard pseudo label。
        # #    pseudo_1 / pseudo_2 仍然保留，只用于后面的无标签对比学习采样 mask。
        # with torch.no_grad():
        #     prob1 = torch.sigmoid(pred1_u)
        #     prob2 = torch.sigmoid(pred2_u)
        #
        #     # 每个样本随机一个 alpha，形状自动广播到 [B, 1, D, H, W]
        #     alpha = torch.rand(
        #         size=(prob1.shape[0], 1, 1, 1, 1),
        #         device=prob1.device,
        #         dtype=prob1.dtype
        #     )
        #
        #     # 不截断，保留 soft pseudo label
        #     pseudo_mix = alpha * prob1 + (1.0 - alpha) * prob2
        #
        #     # 这两个 hard mask 只给后面的 CL 使用，不参与 loss_ps
        #     pseudo_1 = (prob1 > 0.5).float()
        #     pseudo_2 = (prob2 > 0.5).float()
        #
        # loss_ps_1 = self.pseudo_loss_fn(pred1_u, pseudo_mix)
        # loss_ps_2 = self.pseudo_loss_fn(pred2_u, pseudo_mix)
        # loss_ps = 0.5 * (loss_ps_1 + loss_ps_2)
        # loss_ps = loss_ps * (self.max_pseudo_weight * self.sigmoid_rampup(current_epoch, self.ramp_epochs))
        #
        # 3. 对比学习
        loss_cl = torch.tensor(0.0, device=pred1_l.device)
        if self.enable_cl and feats_l is not None:
            loss_cl += 0.5 * (
                        self.cl_loss_fn(feats_l[0], mask_l, True, True) + self.cl_loss_fn(feats_l[1], mask_l, True,
                                                                                          True))

            if current_epoch >= self.warmup:
                loss_cl += 0.5 * (
                            self.cl_loss_fn(feats_u[0], reliable_pseudo, False, False) + self.cl_loss_fn(feats_u[1], reliable_pseudo,
                                                                                                  False, False))

        total_loss = loss_sup + loss_ps + (self.cl_weight * loss_cl if self.enable_cl else 0.0)
        return total_loss, loss_sup, loss_ps, loss_cl
