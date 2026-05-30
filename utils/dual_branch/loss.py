import numpy as np
import torch
import torch.nn as nn

from utils.dual_branch.simple_loss import SparseSliceLoss


class DualBranchLoss(nn.Module):
    """
    基础双分支 CPS 损失，用于切片弱监督半监督训练。
    损失由两部分组成：
      1. 有标签/切片弱标注样本上的真实监督损失。
      2. CPS 交叉伪标签损失，计算区域包括：
         - 无标签样本的全部体素；
         - 切片弱标注样本中的未知区域，也就是 mask_l == ignore_index 的区域。
    注意：
      有标签样本中已经被真实切片标签标注过的体素，只使用真实标签监督。
      伪标签不会作用在这些真实标注体素上，避免伪标签覆盖或干扰真实监督信号。
    """

    def __init__(
        self,
        base_sup_loss,
        pseudo_loss_fn,
        ramp_epochs=50,
        max_pseudo_weight=0.5,
        pseudo_label_mode="hard",
    ):
        super().__init__()
        self.sup_loss_fn = base_sup_loss
        self.pseudo_loss_fn = pseudo_loss_fn
        self.ramp_epochs = ramp_epochs
        self.max_pseudo_weight = max_pseudo_weight
        self.pseudo_label_mode = str(pseudo_label_mode).lower()

    def sigmoid_rampup(self, current, rampup_length):
        """计算伪标签损失的 ramp-up 权重；rampup_length <= 0 时表示关闭 ramp-up。"""
        if rampup_length <= 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def _pseudo_targets(self, pred1, pred2):
        """
        根据两个分支的预测生成伪标签。
        hard:
            分支 1 学习分支 2 产生的 0/1 硬伪标签；
            分支 2 学习分支 1 产生的 0/1 硬伪标签。
        soft:
            将两个分支的概率图随机混合成一张 soft pseudo label；
            两个分支共同学习这张软伪标签。
        """
        prob1 = torch.sigmoid(pred1)
        prob2 = torch.sigmoid(pred2)

        if self.pseudo_label_mode == "soft":
            alpha = torch.rand(
                size=(prob1.shape[0], 1, 1, 1, 1),
                device=prob1.device,
                dtype=prob1.dtype,
            )
            pseudo_mix = alpha * prob1 + (1.0 - alpha) * prob2
            return pseudo_mix, pseudo_mix

        return (prob2 > 0.5).float(), (prob1 > 0.5).float()

    def _pseudo_loss_pair(self, pred1, pred2, valid_mask=None):
        """
        计算两个分支之间的 CPS 伪标签损失。
        valid_mask=None:
            所有体素都参与伪标签损失，用于完全无标签样本。
        valid_mask=(mask_l == ignore_index):
            只有未知体素参与伪标签损失，用于切片弱标注样本。
        """
        with torch.no_grad():
            pseudo_target_1, pseudo_target_2 = self._pseudo_targets(pred1, pred2)

        loss_ps_1 = self.pseudo_loss_fn(pred1, pseudo_target_1, valid_mask=valid_mask)
        loss_ps_2 = self.pseudo_loss_fn(pred2, pseudo_target_2, valid_mask=valid_mask)
        return 0.5 * (loss_ps_1 + loss_ps_2)

    def forward(self, preds_l, mask_l, preds_u, current_epoch, img_l=None):
        pred1_l, pred2_l = preds_l
        pred1_u, pred2_u = preds_u
        # 1. 有标签样本上的真实监督损失：只在真实标注区域计算。
        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l, img_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l, img_l)
        else:
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l)
        loss_sup = 0.5 * (loss_sup_1 + loss_sup_2)
        # 2. 无标签样本上的 CPS 伪标签损失：全图体素参与。
        loss_ps_u = self._pseudo_loss_pair(pred1_u, pred2_u, valid_mask=None)
        # 3. 切片弱标注样本上的 CPS 伪标签损失：只在 mask_l == ignore_index 的未知区域计算。
        ignore_index = getattr(self.sup_loss_fn, "ignore_index", 255)
        labeled_unknown_mask = (mask_l == ignore_index).float()

        if labeled_unknown_mask.sum() > 0:
            loss_ps_l = self._pseudo_loss_pair(pred1_l, pred2_l, valid_mask=labeled_unknown_mask)
            loss_ps = 0.5 * (loss_ps_u + loss_ps_l)
        else:
            # 如果当前 labeled batch 没有未知区域，则不把无标签 CPS 损失额外稀释。
            loss_ps_l = pred1_l.sum() * 0.0
            loss_ps = loss_ps_u
        # 4. 根据 ramp-up / 冷启动设置，对伪标签损失施加当前权重。
        rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        current_pseudo_weight = self.max_pseudo_weight * rampup_weight
        total_loss = loss_sup + current_pseudo_weight * loss_ps

        return total_loss, loss_sup, loss_ps
