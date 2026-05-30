import numpy as np
import torch
import torch.nn as nn

from utils.dual_branch.simple_loss import SparseSliceLoss


class DualBranchLoss(nn.Module):
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
        if rampup_length <= 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def _pseudo_targets(self, pred1_u, pred2_u):
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
        pred1_l, pred2_l = preds_l
        pred1_u, pred2_u = preds_u

        if isinstance(self.sup_loss_fn, SparseSliceLoss):
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l, img_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l, img_l)
        else:
            loss_sup_1 = self.sup_loss_fn(pred1_l, mask_l)
            loss_sup_2 = self.sup_loss_fn(pred2_l, mask_l)
        loss_sup = 0.5 * (loss_sup_1 + loss_sup_2)

        with torch.no_grad():
            pseudo_target_1, pseudo_target_2 = self._pseudo_targets(pred1_u, pred2_u)

        loss_ps_1 = self.pseudo_loss_fn(pred1_u, pseudo_target_1)
        loss_ps_2 = self.pseudo_loss_fn(pred2_u, pseudo_target_2)
        loss_ps = 0.5 * (loss_ps_1 + loss_ps_2)

        rampup_weight = self.sigmoid_rampup(current_epoch, self.ramp_epochs)
        current_pseudo_weight = self.max_pseudo_weight * rampup_weight
        total_loss = loss_sup + current_pseudo_weight * loss_ps

        return total_loss, loss_sup, loss_ps


