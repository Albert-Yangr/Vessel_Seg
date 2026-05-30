import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.dual_branch.simple_loss import SparseSliceLoss


# =========================================================================
# 🌟 创新模块：连通域形态感知对比学习 (保持完全不变，你的核心 Contribution)
# =========================================================================
class ComponentAdaptivePatchContrastiveLoss(nn.Module):
    """
    Component-aware adaptive 2D patch contrastive learning.

    For each sampled vessel anchor, it finds the 2D connected component that contains the anchor on
    the selected slice, builds the component's rectangular bbox, expands it with context, then
    resamples that arbitrary HxW ROI to a fixed resolution. This is intentionally ROI-like: long
    brain-vessel components stay long before resampling, large pulmonary blobs can be fully covered,
    and coronary dot-like components naturally collapse to small local ROIs.
    """

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
        self.max_component_pixels = int(cfg.get("max_component_pixels", 4096))
        self.max_bbox_area_ratio = float(cfg.get("max_bbox_area_ratio", 0.75))
        self.min_resized_fg_pixels = int(cfg.get("min_resized_fg_pixels", 2))
        self.max_flood_iters = int(cfg.get("max_flood_iters", 256))
        self.near_bg_kernel = int(cfg.get("near_bg_kernel", 9))
        self.proto_momentum = float(cfg.get("proto_momentum", 0.95))

        self.register_buffer("vessel_proto", F.normalize(torch.randn(self.feat_dim), dim=0))
        self.register_buffer("bg_proto", F.normalize(torch.randn(self.feat_dim), dim=0))

    def _dynamic_temp_from_margin(self, pos_sim, neg_sim):
        if not self.dynamic_temperature:
            return torch.full_like(pos_sim, self.temperature)

        # Cosine similarity lies in [-1, 1], so a positive-vs-negative margin lies roughly in [0, 2].
        # Higher confidence gets lower temperature; uncertain samples get a softer high temperature.
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

    @torch.no_grad()
    def update_prototypes(self, v_feats, b_feats):
        if v_feats is not None and v_feats.numel() > 0:
            v_mean = F.normalize(v_feats.mean(dim=0), dim=0)
            new_v = F.normalize(
                self.proto_momentum * self.vessel_proto + (1.0 - self.proto_momentum) * v_mean,
                dim=0,
            )
            self.vessel_proto.copy_(new_v)

        if b_feats is not None and b_feats.numel() > 0:
            b_mean = F.normalize(b_feats.mean(dim=0), dim=0)
            new_b = F.normalize(
                self.proto_momentum * self.bg_proto + (1.0 - self.proto_momentum) * b_mean,
                dim=0,
            )
            self.bg_proto.copy_(new_b)

    def _resize_roi(self, feat_patch, mask_patch):
        feat_patch = F.interpolate(
            feat_patch.unsqueeze(0),
            size=(self.roi_size, self.roi_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_patch = F.interpolate(
            mask_patch.float().unsqueeze(0),
            size=(self.roi_size, self.roi_size),
            mode="nearest",
        ).squeeze(0)
        return feat_patch, mask_patch

    def _take_slice(self, feat3d, mask3d, direction, d, h, w):
        if direction == 0:
            return feat3d[:, d, :, :], mask3d[:, d, :, :], h, w
        if direction == 1:
            return feat3d[:, :, h, :], mask3d[:, :, h, :], d, w
        return feat3d[:, :, :, w], mask3d[:, :, :, w], d, h

    def _direction_from_valid_density(self, mask3d, coords):
        valid = mask3d[0] != self.ignore_index
        density_d = valid.sum(dim=(1, 2))
        density_h = valid.sum(dim=(0, 2))
        density_w = valid.sum(dim=(0, 1))
        return torch.argmax(
            torch.stack([density_d[coords[:, 0]], density_h[coords[:, 1]], density_w[coords[:, 2]]], dim=1),
            dim=1,
        )

    def _direction_from_pseudo_shape(self, mask3d, coords):
        fg = mask3d[0] == 1
        density_d = fg.sum(dim=(1, 2))
        density_h = fg.sum(dim=(0, 2))
        density_w = fg.sum(dim=(0, 1))
        stacked = torch.stack([density_d[coords[:, 0]], density_h[coords[:, 1]], density_w[coords[:, 2]]], dim=1)
        dirs = torch.argmax(stacked, dim=1)
        empty = stacked.sum(dim=1) == 0
        if empty.any():
            dirs[empty] = torch.randint(0, 3, (int(empty.sum().item()),), device=mask3d.device)
        return dirs

    def _flood_component_2d(self, fg_local, sy, sx):
        if fg_local.numel() == 0 or not bool(fg_local[sy, sx]):
            return None

        comp = torch.zeros_like(fg_local, dtype=torch.bool)
        comp[sy, sx] = True
        fg4 = fg_local[None, None].float()

        for _ in range(self.max_flood_iters):
            prev = comp
            grown = F.max_pool2d(comp[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0] > 0
            comp = grown & fg_local
            if torch.equal(comp, prev):
                break

        return comp & (fg4[0, 0] > 0)

    def _component_crop(self, feat2d, mask2d, ay, ax):
        _, H, W = feat2d.shape
        if self.search_size and self.search_size > 0:
            half = self.search_size // 2
            y0 = max(0, int(ay) - half)
            y1 = min(H, int(ay) + half + 1)
            x0 = max(0, int(ax) - half)
            x1 = min(W, int(ax) + half + 1)
        else:
            y0, y1, x0, x1 = 0, H, 0, W

        fg_local = mask2d[0, y0:y1, x0:x1] == 1
        sy = int(ay) - y0
        sx = int(ax) - x0
        comp_local = self._flood_component_2d(fg_local, sy, sx)

        if comp_local is None or int(comp_local.sum().item()) < self.min_component_pixels:
            return self._fixed_vessel_crop(feat2d, mask2d, ay, ax)

        component_pixels = int(comp_local.sum().item())
        if component_pixels > self.max_component_pixels:
            return self._fixed_vessel_crop(feat2d, mask2d, ay, ax)

        coords = torch.nonzero(comp_local, as_tuple=False)
        yy0 = y0 + int(coords[:, 0].min().item())
        yy1 = y0 + int(coords[:, 0].max().item()) + 1
        xx0 = x0 + int(coords[:, 1].min().item())
        xx1 = x0 + int(coords[:, 1].max().item()) + 1

        comp_h = yy1 - yy0
        comp_w = xx1 - xx0
        margin = int(round(max(comp_h, comp_w) * self.margin_ratio))
        margin = max(self.min_margin, min(margin, self.max_margin))

        yy0 = max(0, yy0 - margin)
        yy1 = min(H, yy1 + margin)
        xx0 = max(0, xx0 - margin)
        xx1 = min(W, xx1 + margin)

        bbox_area_ratio = ((yy1 - yy0) * (xx1 - xx0)) / max(1, H * W)
        if bbox_area_ratio > self.max_bbox_area_ratio:
            return self._fixed_vessel_crop(feat2d, mask2d, ay, ax)

        feat_patch = feat2d[:, yy0:yy1, xx0:xx1]
        raw_mask = mask2d[:, yy0:yy1, xx0:xx1].clone()

        comp_full = torch.zeros((H, W), dtype=torch.bool, device=mask2d.device)
        comp_full[y0:y1, x0:x1] = comp_local
        comp_crop = comp_full[yy0:yy1, xx0:xx1]

        # Keep only the anchor component as vessel. Other foreground fragments are ignored instead
        # of being treated as positive or negative samples.
        patch_mask = raw_mask.clone()
        other_fg = (patch_mask[0] == 1) & (~comp_crop)
        patch_mask[0, other_fg] = self.ignore_index
        patch_mask[0, comp_crop] = 1

        feat_patch, patch_mask = self._resize_roi(feat_patch, patch_mask)
        if int((patch_mask[0] == 1).sum().item()) < self.min_resized_fg_pixels:
            return self._fixed_vessel_crop(feat2d, mask2d, ay, ax)
        return feat_patch, patch_mask

    def _fixed_vessel_crop(self, feat2d, mask2d, ay, ax):
        _, H, W = feat2d.shape
        size = min(self.base_patch_size, H, W)
        y0 = max(0, min(int(round(ay - size / 2)), H - size))
        x0 = max(0, min(int(round(ax - size / 2)), W - size))
        feat_patch = feat2d[:, y0:y0 + size, x0:x0 + size]
        mask_patch = mask2d[:, y0:y0 + size, x0:x0 + size].clone()
        mask_patch[0, mask_patch[0] == 1] = 1
        return self._resize_roi(feat_patch, mask_patch)

    def _fixed_bg_crop(self, feat2d, mask2d, ay, ax):
        _, H, W = feat2d.shape
        size = min(self.base_patch_size, H, W)
        y0 = max(0, min(int(round(ay - size / 2)), H - size))
        x0 = max(0, min(int(round(ax - size / 2)), W - size))
        feat_patch = feat2d[:, y0:y0 + size, x0:x0 + size]
        mask_patch = mask2d[:, y0:y0 + size, x0:x0 + size].clone()
        mask_patch[0, mask_patch[0] == 1] = self.ignore_index
        return self._resize_roi(feat_patch, mask_patch)

    def sample_and_crop(self, feat, mask, is_gt=True):
        mask = self._resize_mask_to_feat(mask, feat)
        B, _, _, _, _ = feat.shape
        v_patches_feat, v_patches_mask = [], []
        b_patches_feat, b_patches_mask = [], []

        for b in range(B):
            f_b = feat[b]
            m_b = mask[b]
            v_coords = torch.nonzero(m_b[0] == 1, as_tuple=False)
            b_coords = torch.nonzero(m_b[0] == 0, as_tuple=False)

            if v_coords.shape[0] > self.num_patches:
                v_coords = v_coords[torch.randperm(v_coords.shape[0], device=v_coords.device)[:self.num_patches]]
            if b_coords.shape[0] > min(5000, max(self.num_patches, 1)):
                b_coords = b_coords[torch.randperm(b_coords.shape[0], device=b_coords.device)[:5000]]
            if b_coords.shape[0] > self.num_patches:
                b_coords = b_coords[torch.randperm(b_coords.shape[0], device=b_coords.device)[:self.num_patches]]

            if v_coords.numel() > 0:
                v_dirs = self._direction_from_valid_density(m_b,
                                                            v_coords) if is_gt else self._direction_from_pseudo_shape(
                    m_b, v_coords)
            else:
                v_dirs = torch.zeros(0, dtype=torch.long, device=feat.device)
            if b_coords.numel() > 0:
                b_dirs = self._direction_from_valid_density(m_b, b_coords) if is_gt else torch.randint(0, 3, (
                b_coords.shape[0],), device=feat.device)
            else:
                b_dirs = torch.zeros(0, dtype=torch.long, device=feat.device)

            for coord, direction in zip(v_coords.cpu().tolist(), v_dirs.cpu().tolist()):
                d, h, w = coord
                feat2d, mask2d, ay, ax = self._take_slice(f_b, m_b, direction, d, h, w)
                c_f, c_m = self._component_crop(feat2d, mask2d, int(ay), int(ax))
                v_patches_feat.append(c_f.unsqueeze(0))
                v_patches_mask.append(c_m.unsqueeze(0))

            for coord, direction in zip(b_coords.cpu().tolist(), b_dirs.cpu().tolist()):
                d, h, w = coord
                feat2d, mask2d, ay, ax = self._take_slice(f_b, m_b, direction, d, h, w)
                c_f, c_m = self._fixed_bg_crop(feat2d, mask2d, int(ay), int(ax))
                b_patches_feat.append(c_f.unsqueeze(0))
                b_patches_mask.append(c_m.unsqueeze(0))

        return v_patches_feat, v_patches_mask, b_patches_feat, b_patches_mask

    def _class_macro_feats(self, feats_list, masks_list, cls_value):
        if not feats_list:
            return None
        feats = torch.cat(feats_list, dim=0)
        masks = torch.cat(masks_list, dim=0)
        n, c = feats.shape[:2]
        flat_f = feats.reshape(n, c, -1)
        flat_m = masks.reshape(n, -1)
        cls_mask = (flat_m == cls_value).float()
        counts = cls_mask.sum(dim=1, keepdim=True)
        keep = counts.squeeze(1) > 0
        if not keep.any():
            return None
        return torch.bmm(flat_f[keep], cls_mask[keep].unsqueeze(2)).squeeze(2) / (counts[keep] + 1e-6)

    def _near_background_mask(self, v_mask_flat, b_mask_flat, side):
        v2d = v_mask_flat.reshape(-1, 1, side, side).float()
        kernel = max(3, self.near_bg_kernel)
        if kernel % 2 == 0:
            kernel += 1
        near = F.max_pool2d(v2d, kernel_size=kernel, stride=1, padding=kernel // 2).reshape(-1, side * side) > 0
        return b_mask_flat.bool() & near

    def compute_macro_micro_loss(self, v_feats_list, v_masks_list, b_feats_list, b_masks_list):
        device = self.vessel_proto.device
        macro_loss = torch.tensor(0.0, device=device)
        micro_loss = torch.tensor(0.0, device=device)
        valid_macro = 0
        valid_micro = 0

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
            vf = torch.cat(v_feats_list, dim=0)
            vm = torch.cat(v_masks_list, dim=0)
            n, c, h, w = vf.shape
            ff = vf.reshape(n, c, -1)
            mm = vm.reshape(n, -1)
            v_mask = (mm == 1).float()
            b_mask = (mm == 0).float()
            v_count = v_mask.sum(dim=1)
            b_count = b_mask.sum(dim=1)
            keep = (v_count > 0) & (b_count > 0)
            if keep.any():
                f_keep = ff[keep]
                v_keep = v_mask[keep]
                b_keep = b_mask[keep]
                anchor = torch.bmm(f_keep, v_keep.unsqueeze(2)).squeeze(2) / (v_keep.sum(dim=1, keepdim=True) + 1e-6)
                anchor = F.normalize(anchor, dim=1)
                pos_sim = torch.matmul(anchor, self.vessel_proto)

                near_bg = self._near_background_mask(v_keep, b_keep, h).float()
                bg_count = near_bg.sum(dim=1)
                fallback = bg_count <= 0
                if fallback.any():
                    near_bg[fallback] = b_keep[fallback]

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
        if update_proto:
            self.update_prototypes(
                self._class_macro_feats(v_f, v_m, cls_value=1),
                self._class_macro_feats(b_f, b_m, cls_value=0),
            )
        return self.compute_macro_micro_loss(v_f, v_m, b_f, b_m)


# =========================================================================
# 🌟 完全原版的双分支包装器 (100% 还原基线)
# (已剥离所有多余的伪标签置信度魔改，仅保留最基础的截断逻辑)
# =========================================================================
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
            self.cl_loss_fn = ComponentAdaptivePatchContrastiveLoss(cl_cfg)
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
        feat1_l, feat2_l = feats_l
        feat1_u, feat2_u = feats_u

        # ==========================================
        # 1. 监督损失 (回归基础版逻辑)
        # ==========================================
        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            loss_sup = 0.5 * (
                    self.sup_loss_fn(pred1_l, mask_l, img_l)
                    + self.sup_loss_fn(pred2_l, mask_l, img_l)
            )
        else:
            loss_sup = 0.5 * (
                    self.sup_loss_fn(pred1_l, mask_l)
                    + self.sup_loss_fn(pred2_l, mask_l)
            )

        # ==========================================
        # 2. 交叉伪标签损失 (完全退回原始逻辑：简单截断，不带Mask屏蔽)
        # ==========================================
        with torch.no_grad():
            loss_target_1, loss_target_2 = self._pseudo_loss_targets(pred1_u, pred2_u)
            _, reliable_pseudo = self._make_pseudo_and_reliable_masks(pred1_u, pred2_u)

        # 直接传入 pseudo_loss_fn 互相监督 (不带 mask)
        loss_ps_1 = self.pseudo_loss_fn(pred1_u, loss_target_1)
        loss_ps_2 = self.pseudo_loss_fn(pred2_u, loss_target_2)

        rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        loss_ps = 0.5 * (loss_ps_1 + loss_ps_2) * (self.max_pseudo_weight * rampup_weight)
        # ==========================================
        # 2. 混合 Soft 伪标签损失
        # ==========================================
        # with torch.no_grad():
        #     prob1 = torch.sigmoid(pred1_u)
        #     prob2 = torch.sigmoid(pred2_u)
        #
        #     # 每个样本随机一个 alpha，自动广播到 [B, 1, D, H, W]
        #     alpha = torch.rand(
        #         size=(prob1.shape[0], 1, 1, 1, 1),
        #         device=prob1.device,
        #         dtype=prob1.dtype
        #     )
        #
        #     # 不截断，保留 soft pseudo label
        #     pseudo_mix = alpha * prob1 + (1.0 - alpha) * prob2
        #
        #     # 注意：下面这两个 hard pseudo mask 只给无标签对比学习的采样 mask 使用。
        #     # 伪标签损失 loss_ps 不再使用 hard pseudo_1 / pseudo_2。
        #     pseudo_1 = (prob1 > 0.5).float()
        #     pseudo_2 = (prob2 > 0.5).float()
        #
        # # 两个分支共同学习混合后的 soft pseudo label
        # loss_ps_1 = self.pseudo_loss_fn(pred1_u, pseudo_mix)
        # loss_ps_2 = self.pseudo_loss_fn(pred2_u, pseudo_mix)
        #
        # rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        # loss_ps = 0.5 * (loss_ps_1 + loss_ps_2) * (self.max_pseudo_weight * rampup_weight)
        # ==========================================
        # 3. 动态对比学习损失 (你的独家 Contribution)
        # ==========================================
        loss_cl = torch.tensor(0.0, device=pred1_l.device)
        if self.enable_cl and feats_l is not None:
            # 真实标签数据的对比学习
            loss_cl_l1 = self.cl_loss_fn(feat1_l, mask_l, is_gt=True, update_proto=True)
            loss_cl_l2 = self.cl_loss_fn(feat2_l, mask_l, is_gt=True, update_proto=True)
            loss_cl = 0.5 * (loss_cl_l1 + loss_cl_l2)

            # 无标签数据的对比学习
            if current_epoch >= self.warmup:
                # The unlabeled contrastive branch should only sample boxes from high-confidence
                # pseudo regions. A small branch difference alone is insufficient because both
                # decoders can agree around 0.5. Here the pseudo mask keeps only:
                #   1) high-confidence foreground or background,
                #   2) optional branch agreement,
                #   3) optional max branch probability difference.
                loss_cl_u1 = self.cl_loss_fn(feat1_u, reliable_pseudo, is_gt=False, update_proto=False)
                loss_cl_u2 = self.cl_loss_fn(feat2_u, reliable_pseudo, is_gt=False, update_proto=False)
                loss_cl = loss_cl + 0.5 * (loss_cl_u1 + loss_cl_u2)

        total_loss = loss_sup + loss_ps
        if self.enable_cl:
            total_loss = total_loss + self.cl_weight * loss_cl

        return total_loss, loss_sup, loss_ps, loss_cl
