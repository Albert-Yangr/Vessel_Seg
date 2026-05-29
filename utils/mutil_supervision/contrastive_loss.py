import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, max_samples=2048,
                 mode='hard', ignore_index=255,
                 disable_background_clustering=True, **kwargs):
        """
        Args:
            disable_background_clustering: 如果为True (推荐), 则只计算血管作为Anchor的Loss。
                                           允许背景特征自由分布，只要求血管远离背景。
        """
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples
        self.mode = mode
        self.ignore_index = int(ignore_index)
        self.disable_bg_cluster = disable_background_clustering

    def _safe_sample(self, indices, n):
        if len(indices) == 0: return indices
        if len(indices) <= n: return indices
        perm = torch.randperm(len(indices), device=indices.device)[:n]
        return indices[perm]

    def _get_samples(self, preds, target, valid_mask, is_positive):
        """通用采样函数"""
        class_val = 1 if is_positive else 0
        candidate_mask = (target == class_val) & valid_mask
        indices = candidate_mask.nonzero(as_tuple=True)[0]

        if len(indices) < 2: return indices

        # Random 模式或无预测值
        if self.mode == 'random' or preds is None:
            return self._safe_sample(indices, self.max_samples // 2)

        # Hard Mining 模式 (难例挖掘)
        # 对于血管(Pos): 选预测分低的 (难)
        # 对于背景(Neg): 选预测分高的 (难)
        probs_selected = preds[indices]
        descending = not is_positive
        _, sorted_idx = torch.sort(probs_selected, descending=descending)

        k = min(len(sorted_idx), self.max_samples // 2)
        return indices[sorted_idx[:k]]

    def _compute_infonce(self, anchors, keys, labels_anchors, labels_keys):
        """
        计算 InfoNCE Loss
        """
        # L2 归一化
        anchors = F.normalize(anchors, p=2, dim=1)
        keys = F.normalize(keys, p=2, dim=1)

        # 相似度矩阵 [N_anchors, N_keys]
        sim_matrix = torch.matmul(anchors, keys.T) / self.temperature

        # 标签匹配矩阵 (1 表示同类，0 表示异类)
        # [N_anchors, N_keys]
        label_matrix = (labels_anchors.unsqueeze(1) == labels_keys.unsqueeze(0)).float()

        # Mask 处理：如果是同源对比(anchors is keys)，去掉对角线(自己对比自己)
        if anchors is keys:
            logits_mask = torch.scatter(
                torch.ones_like(label_matrix),
                1,
                torch.arange(label_matrix.shape[0]).view(-1, 1).to(label_matrix.device),
                0
            )
        else:
            logits_mask = torch.ones_like(label_matrix)

        # 最终的正样本掩码
        mask = label_matrix * logits_mask

        # LogSumExp 技巧计算分母
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # --- 核心修改：选择哪些 Anchor 参与 Loss 计算 ---
        # 每一行(Anchor) 是否有正样本?
        mask_sum = mask.sum(1)
        valid_rows = (mask_sum > 0)

        # 🔥 如果禁用了背景聚类，我们强制忽略 Label=0 (背景) 的 Anchor
        if self.disable_bg_cluster:
            # 只保留 label 为 1 (血管) 的行
            vessel_anchor_mask = (labels_anchors == 1)
            valid_rows = valid_rows & vessel_anchor_mask

        if valid_rows.sum() == 0:
            return torch.tensor(0.0, device=anchors.device, requires_grad=True)

        # 只计算有效 Anchor 的平均 Loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-6)
        loss = - mean_log_prob_pos[valid_rows].mean()

        return loss

    def forward(self, project_out, target, predict=None):
        """单尺度内部对比"""
        B, C, D, H, W = project_out.shape
        proj_flat = project_out.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        target_flat = target.view(-1)
        valid_mask = (target_flat != self.ignore_index)

        predict_flat = predict.view(-1) if predict is not None else None

        if valid_mask.sum() == 0: return 0.0 * project_out.sum()

        # 1. 采样 Anchor 和 Key
        # 即使 disable_bg_cluster=True，我们也需要采样背景作为负样本(Keys)，
        # 只是不让背景做 Anchor 而已。
        idx_v = self._get_samples(predict_flat, target_flat, valid_mask, is_positive=True)
        idx_b = self._get_samples(predict_flat, target_flat, valid_mask, is_positive=False)

        if len(idx_v) < 2: return 0.0 * project_out.sum()

        # 这里的 indices 包含了血管和背景
        sampled_indices = torch.cat([idx_v, idx_b], dim=0)
        feats = proj_flat[sampled_indices]
        lbls = target_flat[sampled_indices]

        # 计算 Loss (内部会根据 disable_bg_cluster 过滤 Anchor)
        return self._compute_infonce(feats, feats, lbls, lbls)

    def forward_cross_scale(self, feat_high, mask_high, feat_low, mask_low):
        """跨尺度对比"""
        # ... (采样逻辑保持不变，重点是最后调用 _compute_infonce 时会自动应用过滤) ...
        # 为了简洁，这里省略重复采样代码，逻辑与原文件一致
        # 只要 _compute_infonce 改了，这里也会生效

        # 简写采样逻辑复用：
        # ... (此处应保留原本的采样代码) ...

        # 假设 idx_high 和 idx_low 已经采样好
        # return self._compute_infonce(anchors, keys, lbl_anchors, lbl_keys)
        # 为确保完整性，建议保留你原文件的 forward_cross_scale 采样部分
        # 仅 _compute_infonce 方法发生了逻辑改变。

        # (这里为了代码不报错，还是把原采样逻辑贴上)
        B, C, D, H, W = feat_high.shape
        high_flat = feat_high.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        mask_high_flat = mask_high.view(-1)
        valid_high = (mask_high_flat != self.ignore_index)

        idx_h_v = self._safe_sample(((mask_high_flat == 1) & valid_high).nonzero(as_tuple=True)[0],
                                    self.max_samples // 2)
        idx_h_b = self._safe_sample(((mask_high_flat == 0) & valid_high).nonzero(as_tuple=True)[0],
                                    self.max_samples // 2)

        B2, C2, D2, H2, W2 = feat_low.shape
        low_flat = feat_low.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        mask_low_flat = mask_low.view(-1)
        valid_low = (mask_low_flat != self.ignore_index)

        idx_l_v = self._safe_sample(((mask_low_flat == 1) & valid_low).nonzero(as_tuple=True)[0], self.max_samples // 2)
        idx_l_b = self._safe_sample(((mask_low_flat == 0) & valid_low).nonzero(as_tuple=True)[0], self.max_samples // 2)

        if len(idx_h_v) < 1 or len(idx_l_v) < 1: return 0.0 * feat_high.sum()

        idx_high = torch.cat([idx_h_v, idx_h_b], dim=0)
        idx_low = torch.cat([idx_l_v, idx_l_b], dim=0)

        anchors = high_flat[idx_high]
        lbl_anchors = mask_high_flat[idx_high]
        keys = low_flat[idx_low]
        lbl_keys = mask_low_flat[idx_low]

        return self._compute_infonce(anchors, keys, lbl_anchors, lbl_keys)