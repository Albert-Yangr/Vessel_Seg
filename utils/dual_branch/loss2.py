import torch
import torch.nn as nn
import numpy as np
# 引入类型判断
from utils.dual_branch.simple_loss import SparseSliceLoss


class DualBranchLoss(nn.Module):
    def __init__(self, base_sup_loss, pseudo_loss_fn, ramp_epochs=50, max_pseudo_weight=0.5):
        super().__init__()
        self.sup_loss_fn = base_sup_loss
        self.pseudo_loss_fn = pseudo_loss_fn
        self.ramp_epochs = ramp_epochs
        self.max_pseudo_weight = max_pseudo_weight

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0: return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def forward(self, preds_l, mask_l, preds_u, current_epoch, img_l=None):
        """
        新增参数 img_l: 用于 SparseSliceLoss 计算亲和力
        """
        pred1_l, pred2_l = preds_l
        pred1_u, pred2_u = preds_u

        # ==========================================
        # 1. 监督损失 (针对有标签/稀疏标签数据)
        # ==========================================
        # 🌟 逻辑判断：如果使用的是 SparseSliceLoss，则传入 img_l
        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l, img_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l, img_l)
        else:
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l)

        loss_sup = 0.5 * (loss_sup_1 + loss_sup_2)

        # ==========================================
        # 2. 交叉伪标签损失 (针对无标签数据)

        with torch.no_grad():
            prob1 = torch.sigmoid(pred1_u)
            prob2 = torch.sigmoid(pred2_u)

            # 每个样本随机一个 alpha，形状可以自动广播到 [B, 1, D, H, W]
            alpha = torch.rand(
                size=(prob1.shape[0], 1, 1, 1, 1),
                device=prob1.device,
                dtype=prob1.dtype
            )

            # 不截断，保留 soft pseudo label
            pseudo_mix = alpha * prob1 + (1.0 - alpha) * prob2

        loss_ps_1 = self.pseudo_loss_fn(pred1_u, pseudo_mix)
        loss_ps_2 = self.pseudo_loss_fn(pred2_u, pseudo_mix)
        loss_ps = 0.5 * (loss_ps_1 + loss_ps_2)

        # 权重预热
        rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        current_pseudo_weight = self.max_pseudo_weight * rampup_weight

        total_loss = loss_sup + current_pseudo_weight * loss_ps

        return total_loss, loss_sup, loss_ps