import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.mutil_supervision.slice_loss import SparseSliceLoss


# =========================================================================
# 🌟 新增：双层级切片对比学习核心模块
# =========================================================================
class BiLevelContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temp = cfg.temperature
        self.momentum = cfg.momentum
        self.patch_size = tuple(cfg.patch_size)  # (D, H, W) e.g., (1, 16, 16)
        self.num_patches = cfg.num_patches_per_class
        self.num_pixels = cfg.pixels_per_patch
        self.hard_ratio = cfg.hard_neg_ratio

        # 注册全局原型为 Buffer，不参与梯度反向传播，随模型保存
        self.register_buffer("vessel_proto", torch.randn(cfg.feature_dim))
        self.register_buffer("bg_proto", torch.randn(cfg.feature_dim))
        self.vessel_proto = F.normalize(self.vessel_proto, dim=0)
        self.bg_proto = F.normalize(self.bg_proto, dim=0)

    @torch.no_grad()
    def _update_prototypes(self, v_feat, b_feat):
        """EMA 动量更新原型"""
        if v_feat is not None:
            self.vessel_proto = F.normalize(self.momentum * self.vessel_proto + (1 - self.momentum) * v_feat.mean(0),
                                            dim=0)
        if b_feat is not None:
            self.bg_proto = F.normalize(self.momentum * self.bg_proto + (1 - self.momentum) * b_feat.mean(0), dim=0)

    def forward(self, features, masks):
        """
        features: (B, C, D, H, W) 32维特征图
        masks: (B, 1, D, H, W) 0=背景, 1=血管, 255=无标签/忽略
        """
        B, C, D, H, W = features.shape
        loss_patch = torch.tensor(0.0, device=features.device)
        loss_pixel = torch.tensor(0.0, device=features.device)

        # 解析真实物理标签
        valid_mask = (masks != 255).float()
        vessel_mask = (masks == 1).float() * valid_mask
        bg_mask = (masks == 0).float() * valid_mask

        if valid_mask.sum() == 0:
            return loss_patch

        # ==========================================
        # 1. 宏观：Patch-Level 特征提取与对比
        # ==========================================
        # 使用 Average Pooling 提取框特征
        patch_features = F.avg_pool3d(features, kernel_size=self.patch_size, stride=self.patch_size)

        # 判断小框的阵营 (Max Pooling只要框内有1，结果就是1)
        patch_has_vessel = F.max_pool3d(vessel_mask, kernel_size=self.patch_size, stride=self.patch_size) > 0
        patch_is_valid = F.max_pool3d(valid_mask, kernel_size=self.patch_size, stride=self.patch_size) > 0
        patch_is_pure_bg = (patch_has_vessel == 0) & patch_is_valid

        # 铺平空间维度
        feat_flat = patch_features.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)

        # 🌟 核心修复：使用布尔掩码直接提取特征，彻底免疫维度坍塌！
        v_patch_feats = feat_flat[patch_has_vessel.view(-1)]
        b_patch_feats = feat_flat[patch_is_pure_bg.view(-1)]

        # 随机采样框限制数量
        if v_patch_feats.shape[0] > self.num_patches:
            v_patch_feats = v_patch_feats[torch.randperm(v_patch_feats.shape[0])[:self.num_patches]]
        if b_patch_feats.shape[0] > self.num_patches:
            b_patch_feats = b_patch_feats[torch.randperm(b_patch_feats.shape[0])[:self.num_patches]]

        if v_patch_feats.shape[0] == 0: v_patch_feats = None
        if b_patch_feats.shape[0] == 0: b_patch_feats = None

        # 更新原型并计算 Patch Loss (InfoNCE)
        self._update_prototypes(v_patch_feats, b_patch_feats)

        def patch_infonce(feats, pos_proto, neg_proto):
            if feats is None or feats.numel() == 0: return 0.0
            feats = F.normalize(feats, dim=1)
            pos_sim = torch.exp(torch.matmul(feats, pos_proto) / self.temp)
            neg_sim = torch.exp(torch.matmul(feats, neg_proto) / self.temp)
            return -torch.log(pos_sim / (pos_sim + neg_sim)).mean()

        if v_patch_feats is not None:
            loss_patch += patch_infonce(v_patch_feats, self.vessel_proto, self.bg_proto)
        if b_patch_feats is not None:
            loss_patch += patch_infonce(b_patch_feats, self.bg_proto, self.vessel_proto)

        # ==========================================
        # 2. 微观：Intra-Patch 像素级排斥
        # ==========================================
        if v_patch_feats is not None:
            # 使用形态学膨胀找边界 (3D 膨胀)
            dilated_vessel = F.max_pool3d(vessel_mask, kernel_size=3, stride=1, padding=1)
            boundary_bg_mask = (dilated_vessel - vessel_mask) * bg_mask
            normal_bg_mask = bg_mask - boundary_bg_mask

            feat_pixels = features.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)

            num_hard = int(self.num_pixels * self.hard_ratio)
            num_easy = self.num_pixels - num_hard

            # 找到所有血管点、边界背景点、普通背景点
            v_pts = feat_pixels[vessel_mask.squeeze(1) == 1]
            bnd_pts = feat_pixels[boundary_bg_mask.squeeze(1) == 1]
            nrm_pts = feat_pixels[normal_bg_mask.squeeze(1) == 1]

            if v_pts.numel() > 0 and (bnd_pts.numel() > 0 or nrm_pts.numel() > 0):
                # 血管特征取平均作为 Anchor
                v_anchor = F.normalize(v_pts.mean(dim=0, keepdim=True), dim=1)  # (1, C)

                # 混合难例与普通背景作为 Negative
                neg_feats = []
                if bnd_pts.numel() > 0:
                    idx = torch.randperm(bnd_pts.shape[0])[:min(num_hard, bnd_pts.shape[0])]
                    neg_feats.append(bnd_pts[idx])
                if nrm_pts.numel() > 0:
                    idx = torch.randperm(nrm_pts.shape[0])[:min(num_easy, nrm_pts.shape[0])]
                    neg_feats.append(nrm_pts[idx])

                if neg_feats:
                    neg_feats = F.normalize(torch.cat(neg_feats, dim=0), dim=1)  # (N, C)

                    # 算距离：Anchor 和自己的特征更近，和选出来的混合背景疏远
                    pos_sim = torch.exp(torch.matmul(v_anchor, self.vessel_proto) / self.temp)  # 假定靠拢全局血管
                    neg_sim = torch.exp(torch.matmul(v_anchor, neg_feats.T) / self.temp).sum()

                    loss_pixel = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()

        return loss_patch + loss_pixel


# =========================================================================
# 修改原有的 DualBranchLoss，接入对比学习
# =========================================================================
class DualBranchLoss(nn.Module):
    def __init__(self, base_sup_loss, pseudo_loss_fn, ramp_epochs=50, max_pseudo_weight=0.5, cl_cfg=None):
        super().__init__()
        self.sup_loss_fn = base_sup_loss
        self.pseudo_loss_fn = pseudo_loss_fn
        self.ramp_epochs = ramp_epochs
        self.max_pseudo_weight = max_pseudo_weight

        self.uncertainty_threshold = 0.2

        # 🌟 接入双层对比学习组件
        self.cl_cfg = cl_cfg
        self.use_cl = cl_cfg.get("enable", False) if cl_cfg else False
        if self.use_cl:
            self.cl_loss_fn = BiLevelContrastiveLoss(cl_cfg)

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0: return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def forward(self, preds_l, mask_l, preds_u, current_epoch, img_l=None, feats_l=None, feats_u=None):
        pred1_l, pred2_l = preds_l
        pred1_u, pred2_u = preds_u

        # 1. 监督损失
        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l, img_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l, img_l)
        else:
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l)

        loss_sup = 0.5 * (loss_sup_1 + loss_sup_2)

        # 2. 伪标签损失与质检过滤
        with torch.no_grad():
            prob1_u = torch.sigmoid(pred1_u)
            prob2_u = torch.sigmoid(pred2_u)
            pseudo_1 = (prob1_u > 0.5).float()
            pseudo_2 = (prob2_u > 0.5).float()

            diff = torch.abs(prob1_u - prob2_u)
            reliable_mask = (diff < self.uncertainty_threshold).float()

        loss_ps_1 = self.pseudo_loss_fn(pred1_u, pseudo_2, mask=reliable_mask)
        loss_ps_2 = self.pseudo_loss_fn(pred2_u, pseudo_1, mask=reliable_mask)
        loss_ps = 0.5 * (loss_ps_1 + loss_ps_2)

        rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        current_pseudo_weight = self.max_pseudo_weight * rampup_weight

        # 3. 🌟 计算对比学习损失
        loss_cl = torch.tensor(0.0, device=pred1_l.device)
        if self.use_cl and feats_l is not None:
            # 仅在预热期之后启用惩罚，预热期只收集原型
            feat1_l, feat2_l = feats_l

            # 真实标签的对比学习 (最稳定)
            cl_loss_1 = self.cl_loss_fn(feat1_l, mask_l)
            cl_loss_2 = self.cl_loss_fn(feat2_l, mask_l)

            # 伪标签的对比学习 (用过滤后的强一致性掩码，极其严谨！)
            if feats_u is not None:
                feat1_u, feat2_u = feats_u
                # 将不可靠区域置为 255 (忽略)
                pseudo_mask = pseudo_1.clone()
                pseudo_mask[reliable_mask == 0] = 255
                cl_loss_u = self.cl_loss_fn(feat1_u, pseudo_mask) + self.cl_loss_fn(feat2_u, pseudo_mask)
            else:
                cl_loss_u = 0.0

            if current_epoch >= self.cl_cfg.warmup_epochs:
                loss_cl = (cl_loss_1 + cl_loss_2 + cl_loss_u) * self.cl_cfg.weight

        total_loss = loss_sup + current_pseudo_weight * loss_ps + loss_cl

        return total_loss, loss_sup, loss_ps, loss_cl