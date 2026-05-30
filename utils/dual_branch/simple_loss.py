import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSliceLoss(nn.Module):
    """
    普通切片/体素监督损失。

    输入：
        preds:
            模型输出 logits，形状通常为 [B, 1, D, H, W]。
        target:
            标签，0=背景，1=血管，ignore_index=忽略区域。

    计算：
        只在 target != ignore_index 的有效区域计算 BCE + Dice。
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, ignore_index=255):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.bce_func = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, target):
        valid_mask = (target != self.ignore_index).float()

        target_clean = target.clone()
        target_clean[target == self.ignore_index] = 0
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


class SimplePseudoLoss(nn.Module):
    """
    无标签伪标签损失。

    输入：
        preds:
            当前分支对无标签图像的 logits。
        target:
            伪标签，可以是 hard 0/1，也可以是 soft probability map。

    当前配置中 dice_weight=0.9, ce_weight=0.1，因此 Dice 是主项，BCE 是辅助项。
    """

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


class SparseSliceLoss(nn.Module):
    """
    切片弱监督损失。

    适用场景：
        3D 图像中只有少数 2D 切片被标注，未标注区域用 ignore_index=255 表示。

    核心逻辑：
        1. 监督项：只在 slice_label != ignore_index 的区域计算 BCE + Dice。
        2. affinity 项：鼓励原图灰度相近的相邻体素预测也相近。
        3. TV 项：对预测概率做全变分平滑，抑制噪点。

    当前 dual_train.yaml 中 affinity_weight=0, tv_weight=0，因此实际只启用
    标注切片上的 BCE + Dice。
    """

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        affinity_weight=10.0,
        tv_weight=0.1,
        ignore_index=255,
        **kwargs,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.affinity_weight = float(affinity_weight)
        self.tv_weight = float(tv_weight)
        self.ignore_index = int(ignore_index)

        if kwargs:
            print(f"SparseSliceLoss ignored kwargs: {list(kwargs.keys())}")

    def forward(self, pred_logits, slice_label, images):
        """
        pred_logits:
            模型输出 logits。
        slice_label:
            稀疏切片标签，0=背景，1=血管，255=未知区域。
        images:
            原图，用于可选 affinity loss。
        """
        valid_mask = (slice_label != self.ignore_index).float()
        target = (slice_label == 1).float()

        bce_pixel = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")

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
        loss_aff = self.compute_affinity_loss(probs, images, sigma=1.0) if self.affinity_weight > 0 else 0.0
        loss_tv = self.compute_tv_loss(probs) if self.tv_weight > 0 else 0.0

        return (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.affinity_weight * loss_aff
            + self.tv_weight * loss_tv
        )

    def compute_affinity_loss(self, probs, images, sigma=1.0):
        """
        3D affinity loss。

        如果相邻体素在原图中灰度接近，则它们的预测概率也应接近。
        """
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
        """
        Total Variation loss。

        直接惩罚相邻体素预测概率的剧烈变化，使预测更平滑。
        """
        batch_size = probs.size(0)
        h_x, w_x, d_x = probs.shape[2], probs.shape[3], probs.shape[4]

        h_tv = torch.pow((probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :]), 2).sum()
        w_tv = torch.pow((probs[:, :, :, 1:, :] - probs[:, :, :, :-1, :]), 2).sum()
        d_tv = torch.pow((probs[:, :, :, :, 1:] - probs[:, :, :, :, :-1]), 2).sum()

        count = batch_size * h_x * w_x * d_x
        return 2 * (h_tv + w_tv + d_tv) / count
