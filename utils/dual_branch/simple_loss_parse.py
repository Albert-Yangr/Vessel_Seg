import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSliceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, ignore_index=255):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.bce_func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, target):
        valid_mask = (target != self.ignore_index).float()
        target_clean = target.clone()
        target_clean[target == self.ignore_index] = 0

        # 🌟 【核心修复】：将主干血管 (2) 在强监督中重新视为普通血管 (1)
        target_clean[target == 2] = 1

        target_clean = target_clean.float()

        loss_bce = 0.0
        if self.bce_weight > 0:
            bce_pixel = self.bce_func(preds, target_clean)
            if valid_mask.sum() > 0:
                loss_bce = (bce_pixel * valid_mask).sum() / valid_mask.sum()
            else:
                loss_bce = torch.tensor(0.0, device=preds.device, requires_grad=True)

        loss_dice = 0.0
        if self.dice_weight > 0:
            pred_prob = torch.sigmoid(preds)
            intersection = (pred_prob * target_clean * valid_mask).sum()
            denominator = (pred_prob * valid_mask).sum() + (target_clean * valid_mask).sum()
            loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class SparseSliceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, affinity_weight=10.0, tv_weight=0.1, ignore_index=255, **kwargs):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.affinity_weight = float(affinity_weight)
        self.tv_weight = float(tv_weight)
        self.ignore_index = int(ignore_index)

    def forward(self, pred_logits, slice_label, images):
        valid_mask = (slice_label != self.ignore_index).float()

        # 🌟 【核心修复】：标签为 1 (细支) 和 2 (主干) 的区域，统统算作正确的前景 Target
        target = ((slice_label == 1) | (slice_label == 2)).float()

        bce_pixel = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        if valid_mask.sum() > 0:
            loss_bce = (bce_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            pred_probs = torch.sigmoid(pred_logits)
            intersection = (pred_probs * target * valid_mask).sum()
            denominator = ((pred_probs + target) * valid_mask).sum()
            loss_dice = 1.0 - (2.0 * intersection) / (denominator + 1e-5)
        else:
            loss_bce = torch.tensor(0.0, device=pred_logits.device)
            loss_dice = torch.tensor(0.0, device=pred_logits.device)

        probs = torch.sigmoid(pred_logits)
        loss_aff = self.compute_affinity_loss(probs, images, sigma=1.0)
        loss_tv = self.compute_tv_loss(probs)

        return (self.bce_weight * loss_bce) + \
            (self.dice_weight * loss_dice) + \
            (self.affinity_weight * loss_aff) + \
            (self.tv_weight * loss_tv)

    def compute_affinity_loss(self, probs, images, sigma=1.0):
        loss_sum = 0.0
        for dim in [2, 3, 4]:
            img_curr = torch.narrow(images, dim, 0, images.size(dim) - 1)
            img_next = torch.narrow(images, dim, 1, images.size(dim) - 1)
            prob_curr = torch.narrow(probs, dim, 0, probs.size(dim) - 1)
            prob_next = torch.narrow(probs, dim, 1, probs.size(dim) - 1)
            diff_img = torch.abs(img_curr - img_next)
            weight = torch.exp(-diff_img / sigma)
            diff_prob = torch.pow(prob_curr - prob_next, 2)
            loss_sum += (weight * diff_prob).mean()
        return loss_sum

    def compute_tv_loss(self, probs):
        batch_size = probs.size(0)
        h_x, w_x, d_x = probs.shape[2], probs.shape[3], probs.shape[4]
        h_tv = torch.pow((probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :]), 2).sum()
        w_tv = torch.pow((probs[:, :, :, 1:, :] - probs[:, :, :, :-1, :]), 2).sum()
        d_tv = torch.pow((probs[:, :, :, :, 1:] - probs[:, :, :, :, :-1]), 2).sum()
        count = batch_size * h_x * w_x * d_x
        return 2 * (h_tv + w_tv + d_tv) / count


# 🌟 必须加上这个，否则伪标签模块找不到它
class SimplePseudoLoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=0.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.bce_func = nn.BCEWithLogitsLoss()

    def forward(self, preds, target):
        target = target.float()
        loss_dice = 0.0
        if self.dice_weight > 0:
            pred_prob = torch.sigmoid(preds)
            intersection = (pred_prob * target).sum()
            denominator = pred_prob.sum() + target.sum()
            loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)

        loss_ce = 0.0
        if self.ce_weight > 0:
            loss_ce = self.bce_func(preds, target)

        return self.dice_weight * loss_dice + self.ce_weight * loss_ce