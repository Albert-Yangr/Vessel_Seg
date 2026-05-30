import numpy as np
import torch
import torch.nn as nn

from utils.dual_branch.simple_loss import SparseSliceLoss


class DualBranchLoss(nn.Module):
    """
    基础双分支 CPS 总损失。

    这个类只负责把两部分损失合起来：
    1. 有标签/弱监督数据上的监督损失 loss_sup。
    2. 无标签数据上的交叉伪标签损失 loss_ps。

    注意：
    - 这里不包含对比学习。
    - 模型必须在训练时返回两个分支的预测结果: (pred1, pred2)。
    - pseudo_label_mode 控制伪标签形式：
        hard: 传统 CPS，两个分支互相使用 0/1 hard pseudo label 监督。
        soft: 两个分支概率图随机混合，生成 soft pseudo label，两分支共同学习。
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
        """
        伪标签损失权重的 ramp-up 函数。

        rampup_length <= 0 时表示关闭预热，伪标签损失从第 0 轮开始直接使用
        max_pseudo_weight。这个设置适合“冷启动”实验。
        """
        if rampup_length <= 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def _pseudo_targets(self, pred1_u, pred2_u):
        """
        根据两个分支在无标签图像上的预测，生成伪标签监督目标。

        输入：
            pred1_u, pred2_u: 两个分支对无标签图像的 logits。

        输出：
            pseudo_target_1: 用来监督分支 1 的伪标签。
            pseudo_target_2: 用来监督分支 2 的伪标签。

        hard 模式：
            分支 1 学习分支 2 的二值伪标签，分支 2 学习分支 1 的二值伪标签。

        soft 模式：
            不做 0.5 阈值截断，而是将两个分支概率图随机混合成 soft pseudo label。
        """
        prob1 = torch.sigmoid(pred1_u)
        prob2 = torch.sigmoid(pred2_u)

        if self.pseudo_label_mode == "soft":
            alpha = torch.rand(
                size=(prob1.shape[0], 1, 1, 1, 1),
                device=prob1.device,
                dtype=prob1.dtype,
            )
            pseudo_mix = alpha * prob1 + (1.0 - alpha) * prob2
            return pseudo_mix, pseudo_mix

        return (prob2 > 0.5).float(), (prob1 > 0.5).float()

    def forward(self, preds_l, mask_l, preds_u, current_epoch, img_l=None):
        """
        计算一个 batch 的总损失。

        preds_l:
            有标签图像的两个分支预测，形如 (pred1_l, pred2_l)。
        mask_l:
            有标签/弱监督标签。
        preds_u:
            无标签图像的两个分支预测，形如 (pred1_u, pred2_u)。
        current_epoch:
            当前 epoch，用于 ramp-up。
        img_l:
            原图。当监督损失是 SparseSliceLoss 时，需要原图计算可选的
            affinity loss；当前配置里 affinity_weight=0，也可以正常传入。
        """
        pred1_l, pred2_l = preds_l
        pred1_u, pred2_u = preds_u

        # 1. 有标签/弱监督分支的监督损失。
        # 两个 decoder 都要被真实标签约束，最后取平均。
        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l, img_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l, img_l)
        else:
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l)
        loss_sup = 0.5 * (loss_sup_1 + loss_sup_2)

        # 2. 无标签分支的伪标签损失。
        # 伪标签由模型自身生成，不需要梯度回传到生成伪标签的过程。
        with torch.no_grad():
            pseudo_target_1, pseudo_target_2 = self._pseudo_targets(pred1_u, pred2_u)

        loss_ps_1 = self.pseudo_loss_fn(pred1_u, pseudo_target_1)
        loss_ps_2 = self.pseudo_loss_fn(pred2_u, pseudo_target_2)
        loss_ps = 0.5 * (loss_ps_1 + loss_ps_2)

        # 3. 伪标签权重。
        # ramp_epochs=0 时 rampup_weight=1，直接使用 max_pseudo_weight。
        rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        current_pseudo_weight = self.max_pseudo_weight * rampup_weight
        total_loss = loss_sup + current_pseudo_weight * loss_ps

        return total_loss, loss_sup, loss_ps

