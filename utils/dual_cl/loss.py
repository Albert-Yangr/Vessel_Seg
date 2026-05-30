import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils.dual_branch.simple_loss import SparseSliceLoss


class SliceContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feat_dim = cfg.get("feat_dim", 32)
        self.temperature = cfg.get("temperature", 0.1)
        self.num_patches = cfg.get("num_patches", 16)
        self.patch_size = cfg.get("patch_size", 16)

        # 🌟 EMA 全局原型
        self.register_buffer("vessel_proto", F.normalize(torch.randn(self.feat_dim), dim=0))
        self.register_buffer("bg_proto", F.normalize(torch.randn(self.feat_dim), dim=0))

    def update_prototypes(self, v_feats, b_feats, momentum=0.95):
        """只使用高纯度的特征来更新原型，严格保证DDP同步"""
        if v_feats is not None and v_feats.numel() > 0:
            v_mean = v_feats.mean(dim=0).detach()
            # 🚀 必须使用 .copy_() 原地修改，否则 DDP 会失效！
            new_v = F.normalize(momentum * self.vessel_proto + (1 - momentum) * v_mean, dim=0)
            self.vessel_proto.copy_(new_v)

        if b_feats is not None and b_feats.numel() > 0:
            b_mean = b_feats.mean(dim=0).detach()
            new_b = F.normalize(momentum * self.bg_proto + (1 - momentum) * b_mean, dim=0)
            self.bg_proto.copy_(new_b)

    def safe_crop(self, feat, mask, center, sd, sh, sw):
        D, H, W = feat.shape[1:]
        d, h, w = center

        ds = max(0, min(d - sd // 2, D - sd))
        hs = max(0, min(h - sh // 2, H - sh))
        ws = max(0, min(w - sw // 2, W - sw))

        c_feat = feat[:, ds:ds + sd, hs:hs + sh, ws:ws + sw]
        c_mask = mask[:, ds:ds + sd, hs:hs + sh, ws:ws + sw]
        return c_feat, c_mask

    def sample_and_crop(self, feat, mask, is_gt=True):
        B, C, D, H, W = feat.shape
        v_patches_feat, v_patches_mask = [], []
        b_patches_feat, b_patches_mask = [], []

        for b in range(B):
            f_b = feat[b]
            m_b = mask[b]

            # 1. GPU 原生提取坐标
            v_coords = torch.nonzero(m_b[0] == 1, as_tuple=False)

            num_rand = 5000
            rand_d = torch.randint(0, D, (num_rand,), device=m_b.device)
            rand_h = torch.randint(0, H, (num_rand,), device=m_b.device)
            rand_w = torch.randint(0, W, (num_rand,), device=m_b.device)
            is_bg = (m_b[0, rand_d, rand_h, rand_w] == 0)
            b_coords = torch.stack([rand_d[is_bg], rand_h[is_bg], rand_w[is_bg]], dim=1)

            # 在 GPU 上完成随机打乱并截断
            if len(v_coords) > self.num_patches:
                rand_idx = torch.randperm(len(v_coords), device=m_b.device)[:self.num_patches]
                v_coords = v_coords[rand_idx]
            if len(b_coords) > self.num_patches:
                rand_idx = torch.randperm(len(b_coords), device=m_b.device)[:self.num_patches]
                b_coords = b_coords[rand_idx]

            # ---------------------------------------------------------------------
            # 🚀 GPU 自适应决策方向
            # ---------------------------------------------------------------------
            v_dirs = torch.zeros(len(v_coords), dtype=torch.long, device=m_b.device)
            b_dirs = torch.zeros(len(b_coords), dtype=torch.long, device=m_b.device)

            if is_gt:
                valid_mask = (m_b[0] != 255)
                density_d = valid_mask.sum(dim=(1, 2))  # (D,)
                density_h = valid_mask.sum(dim=(0, 2))  # (H,)
                density_w = valid_mask.sum(dim=(0, 1))  # (W,)

                if len(v_coords) > 0:
                    c_d = density_d[v_coords[:, 0]]
                    c_h = density_h[v_coords[:, 1]]
                    c_w = density_w[v_coords[:, 2]]
                    stacked_v = torch.stack([c_d, c_h, c_w], dim=1)
                    v_dirs = torch.argmax(stacked_v, dim=1)

                if len(b_coords) > 0:
                    c_d = density_d[b_coords[:, 0]]
                    c_h = density_h[b_coords[:, 1]]
                    c_w = density_w[b_coords[:, 2]]
                    stacked_b = torch.stack([c_d, c_h, c_w], dim=1)
                    b_dirs = torch.argmax(stacked_b, dim=1)
            else:
                if len(v_coords) > 0:
                    v_dirs = torch.randint(0, 3, (len(v_coords),), device=m_b.device)
                if len(b_coords) > 0:
                    b_dirs = torch.randint(0, 3, (len(b_coords),), device=m_b.device)

            # ---------------------------------------------------------------------
            # 全局唯一同步点
            # ---------------------------------------------------------------------
            if len(v_coords) > 0:
                v_data = torch.cat([v_coords, v_dirs.unsqueeze(1)], dim=1).cpu().tolist()
            else:
                v_data = []

            if len(b_coords) > 0:
                b_data = torch.cat([b_coords, b_dirs.unsqueeze(1)], dim=1).cpu().tolist()
            else:
                b_data = []

            # =====================================================================
            # 🚀 修复核心：消除不同切面方向带来的维度异构，将其统一为 [1, C, 16, 16]
            # =====================================================================
            for d, h, w, direction in v_data:
                if direction == 0:
                    sd, sh, sw = 1, self.patch_size, self.patch_size
                elif direction == 1:
                    sd, sh, sw = self.patch_size, 1, self.patch_size
                else:
                    sd, sh, sw = self.patch_size, self.patch_size, 1

                c_f, c_m = self.safe_crop(f_b, m_b, (d, h, w), sd, sh, sw)

                # 抹平切面差异
                if direction == 0:
                    c_f, c_m = c_f.squeeze(1), c_m.squeeze(1)
                elif direction == 1:
                    c_f, c_m = c_f.squeeze(2), c_m.squeeze(2)
                else:
                    c_f, c_m = c_f.squeeze(3), c_m.squeeze(3)

                # 添加合并维度 (Batch Dim)
                v_patches_feat.append(c_f.unsqueeze(0))
                v_patches_mask.append(c_m.unsqueeze(0))

            for d, h, w, direction in b_data:
                if direction == 0:
                    sd, sh, sw = 1, self.patch_size, self.patch_size
                elif direction == 1:
                    sd, sh, sw = self.patch_size, 1, self.patch_size
                else:
                    sd, sh, sw = self.patch_size, self.patch_size, 1

                c_f, c_m = self.safe_crop(f_b, m_b, (d, h, w), sd, sh, sw)

                # 抹平切面差异
                if direction == 0:
                    c_f, c_m = c_f.squeeze(1), c_m.squeeze(1)
                elif direction == 1:
                    c_f, c_m = c_f.squeeze(2), c_m.squeeze(2)
                else:
                    c_f, c_m = c_f.squeeze(3), c_m.squeeze(3)

                # 添加合并维度 (Batch Dim)
                b_patches_feat.append(c_f.unsqueeze(0))
                b_patches_mask.append(c_m.unsqueeze(0))

        return v_patches_feat, v_patches_mask, b_patches_feat, b_patches_mask

    # =========================================================================
    # 🌟 核心优化 1：完全向量化的宏观特征提取
    # 彻底告别 for 循环，将所有小框视为一个统一的 3D 张量进行矩阵计算
    # =========================================================================
    def _get_batched_macro_feats(self, feats_list, masks_list):
        if not feats_list:
            return None

        # 堆叠！比如将 16个 [1, C, 16, 16] 变成 [16, C, 16, 16]
        F_t = torch.cat(feats_list, dim=0)
        M_t = torch.cat(masks_list, dim=0)

        N, C = F_t.shape[:2]
        # 展平空间维度 -> [16, C, 256] 和 [16, 256]
        F_flat = F_t.view(N, C, -1)
        M_flat = M_t.view(N, -1)

        # 构建掩码：找到不是 255 的有效像素 (1为有效，0为污染)
        valid_mask = (M_flat != 255).float()
        valid_count = valid_mask.sum(dim=1, keepdim=True)  # 每个框里有效像素的数量 [N, 1]

        # 过滤掉那些“整个框全被 255 污染”的极端无效框
        valid_idx = (valid_count.squeeze(1) > 0)
        if not valid_idx.any():
            return None

        F_valid = F_flat[valid_idx]  # 取出有效的框 [K, C, 256]
        valid_mask = valid_mask[valid_idx].unsqueeze(2)  # [K, 256, 1]
        valid_count = valid_count[valid_idx]  # [K, 1]

        # 🔥 魔法：批量矩阵乘法 (BMM)
        # 用 F_valid [K, C, 256] 乘以 valid_mask [K, 256, 1]
        # 这就相当于只把有效像素加起来，然后除以总数量，得到均值！数学本质与之前一模一样！
        macro_feats = torch.bmm(F_valid, valid_mask).squeeze(2) / valid_count  # -> [K, C]

        return macro_feats

    # =========================================================================
    # 🌟 核心优化 2：微观极限排斥批处理 (内核阻塞终结者)
    # =========================================================================
    def compute_macro_micro_loss(self, v_feats_list, v_masks_list, b_feats_list, b_masks_list):
        macro_loss = torch.tensor(0.0, device=self.vessel_proto.device)
        micro_loss = torch.tensor(0.0, device=self.vessel_proto.device)
        valid_macro = 0
        valid_micro = 0

        # --- 宏观 InfoNCE (批处理版) ---
        v_macro = self._get_batched_macro_feats(v_feats_list, v_masks_list)
        if v_macro is not None:
            v_macro_norm = F.normalize(v_macro, dim=1)  # [K, C]
            pos_sim = torch.exp(torch.matmul(v_macro_norm, self.vessel_proto) / self.temperature)
            neg_sim = torch.exp(torch.matmul(v_macro_norm, self.bg_proto) / self.temperature)
            macro_loss = macro_loss + (-torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean())
            valid_macro += 1

        b_macro = self._get_batched_macro_feats(b_feats_list, b_masks_list)
        if b_macro is not None:
            b_macro_norm = F.normalize(b_macro, dim=1)
            pos_sim = torch.exp(torch.matmul(b_macro_norm, self.bg_proto) / self.temperature)
            neg_sim = torch.exp(torch.matmul(b_macro_norm, self.vessel_proto) / self.temperature)
            macro_loss = macro_loss + (-torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean())
            valid_macro += 1

        # --- 微观极限排斥 (批处理版) ---
        if v_feats_list:
            # 直接将所有血管框堆叠
            V_f = torch.cat(v_feats_list, dim=0)  # [N, C, H, W]
            V_m = torch.cat(v_masks_list, dim=0)  # [N, 1, H, W]

            N, C = V_f.shape[:2]
            V_f_flat = V_f.view(N, C, -1)  # [N, C, 256]
            V_m_flat = V_m.view(N, -1)  # [N, 256]

            # 生成 0/1 掩码
            v_mask = (V_m_flat == 1).float()  # [N, 256]
            b_mask = (V_m_flat == 0).float()  # [N, 256]

            v_count = v_mask.sum(dim=1)
            b_count = b_mask.sum(dim=1)

            # 只对那些“既有血管，又有背景”的重叠边界框进行排斥计算
            valid_micro_idx = (v_count > 0) & (b_count > 0)

            if valid_micro_idx.any():
                F_micro = V_f_flat[valid_micro_idx]  # [K, C, 256]
                v_m_micro = v_mask[valid_micro_idx].unsqueeze(2)  # [K, 256, 1]
                b_m_micro = b_mask[valid_micro_idx]  # [K, 256]
                v_c_micro = v_count[valid_micro_idx].unsqueeze(1)  # [K, 1]

                # 1. 批处理计算 Local Anchor (局部血管点特征的平均值)
                anchor_unnorm = torch.bmm(F_micro, v_m_micro).squeeze(2) / v_c_micro  # [K, C]
                local_anchor = F.normalize(anchor_unnorm, dim=1)  # [K, C]

                # 2. 与全局原型的相似度 (正样本拉近)
                pos_sim = torch.exp(torch.matmul(local_anchor, self.vessel_proto) / self.temperature)  # [K]

                # 3. 与局部背景的相似度排斥 (🔥 难例挖掘核心算法重写)
                F_micro_norm = F.normalize(F_micro, dim=1)  # [K, C, 256]

                # 🛠️ 关键修复 1：给血管锚点加上 .detach()，防止其被巨大的背景梯度拽飞
                local_anchor_detached = local_anchor.detach()

                # 用 detached 的 anchor 瞬间与这 K 个框里的所有 256 个点分别计算余弦相似度
                sim_matrix = torch.bmm(local_anchor_detached.unsqueeze(1), F_micro_norm).squeeze(1)  # [K, 256]
                exp_sim = torch.exp(sim_matrix / self.temperature)

                # 🛠️ 关键修复 2：计算真实背景像素数量，进行均值归一化
                b_count_micro = b_m_micro.sum(dim=1)  # 获取每个框内的实际背景像素数 [K]

                # 累加背景排斥力后，务必除以数量！加上 1e-8 防止除 0 报错
                neg_sim = (exp_sim * b_m_micro).sum(dim=1) / (b_count_micro + 1e-8)  # -> [K]

                # 4. 直接平均合并所有 K 个有效框的微观对比损失
                micro_loss = micro_loss + (-torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean())
                valid_micro += 1

        total = (macro_loss / max(1, valid_macro)) + (micro_loss / max(1, valid_micro))
        return total

    def forward(self, feat, mask, is_gt=True, update_proto=False):
        v_f, v_m, b_f, b_m = self.sample_and_crop(feat, mask, is_gt=is_gt)

        if update_proto:
            # 更新原型依然保持原汁原味的宏观提取特征
            v_gap = self._get_batched_macro_feats(v_f, v_m)
            b_gap = self._get_batched_macro_feats(b_f, b_m)
            self.update_prototypes(v_gap, b_gap)

        return self.compute_macro_micro_loss(v_f, v_m, b_f, b_m)


class DualBranchLoss(nn.Module):
    def __init__(self, base_sup_loss, pseudo_loss_fn, cl_cfg, ramp_epochs=50, max_pseudo_weight=0.5,
                 pseudo_label_mode="hard"):
        super().__init__()
        self.sup_loss_fn = base_sup_loss
        self.pseudo_loss_fn = pseudo_loss_fn
        self.ramp_epochs = ramp_epochs
        self.max_pseudo_weight = max_pseudo_weight
        self.pseudo_label_mode = str(pseudo_label_mode).lower()

        self.cl_cfg = cl_cfg
        self.enable_cl = cl_cfg.get("enable", False) if cl_cfg else False
        if self.enable_cl:
            self.cl_loss_fn = SliceContrastiveLoss(cl_cfg)
            self.cl_weight = cl_cfg.get("weight", 0.1)
            self.warmup = cl_cfg.get("warmup_epochs", 20)

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length <= 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def _pseudo_targets(self, pred1_u, pred2_u):
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

        diff = torch.abs(prob1 - prob2)
        reliable_mask = (diff < 0.2).float()
        return loss_target_1, loss_target_2, pseudo_1, pseudo_2, reliable_mask

    def forward(self, preds_l, mask_l, preds_u, feats_l, feats_u, current_epoch, img_l=None):
        pred1_l, pred2_l = preds_l
        pred1_u, pred2_u = preds_u
        feat1_l, feat2_l = feats_l
        feat1_u, feat2_u = feats_u

        # 1. 监督 Loss
        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l, img_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l, img_l)
        else:
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l)
        loss_sup = 0.5 * (loss_sup_1 + loss_sup_2)

        # 2. 伪标签 Loss
        with torch.no_grad():
            loss_target_1, loss_target_2, pseudo_1, pseudo_2, reliable_mask = self._pseudo_targets(pred1_u, pred2_u)

        loss_ps_1 = self.pseudo_loss_fn(pred1_u, loss_target_1)
        loss_ps_2 = self.pseudo_loss_fn(pred2_u, loss_target_2)
        loss_ps = 0.5 * (loss_ps_1 + loss_ps_2)

        # 3. 对比学习 Loss (渐进式)
        loss_cl = torch.tensor(0.0, device=pred1_l.device)
        if self.enable_cl:
            loss_cl_l1 = self.cl_loss_fn(feat1_l, mask_l, is_gt=True, update_proto=True)
            loss_cl_l2 = self.cl_loss_fn(feat2_l, mask_l, is_gt=True, update_proto=True)
            loss_cl = 0.5 * (loss_cl_l1 + loss_cl_l2)

            if current_epoch >= self.warmup:
                p_mask_1 = pseudo_1.clone()
                p_mask_1[reliable_mask == 0] = 255
                p_mask_2 = pseudo_2.clone()
                p_mask_2[reliable_mask == 0] = 255

                loss_cl_u1 = self.cl_loss_fn(feat1_u, p_mask_1, is_gt=False, update_proto=False)
                loss_cl_u2 = self.cl_loss_fn(feat2_u, p_mask_2, is_gt=False, update_proto=False)
                loss_cl += 0.5 * (loss_cl_u1 + loss_cl_u2)

        rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        total_loss = loss_sup + (self.max_pseudo_weight * rampup_weight * loss_ps)

        if self.enable_cl:
            total_loss += self.cl_weight * loss_cl

        return total_loss, loss_sup, loss_ps, loss_cl
