import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSliceLoss(nn.Module):
    """
    简化的切片损失：只包含 Dice 和 BCE。
    支持 ignore_index (例如 255)，在计算时会自动忽略该区域。
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, ignore_index=255):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        # BCEWithLogitsLoss 自带 sigmoid，且数值稳定
        self.bce_func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, target):
        """
        preds: (B, 1, D, H, W) Logits (未经过 Sigmoid)
        target: (B, 1, D, H, W) 0=背景, 1=前景, 255=忽略
        """
        # 生成有效区域掩码 (不等于 ignore_index 的地方为 1，否则为 0)
        valid_mask = (target != self.ignore_index).float()

        # 将 target 中的 ignore 区域临时设为 0，防止计算出错（反正会被 mask 乘掉）
        target_clean = target.clone()
        target_clean[target == self.ignore_index] = 0
        target_clean = target_clean.float()

        # --- 1. BCE Loss ---
        loss_bce = 0.0
        if self.bce_weight > 0:
            # 计算逐像素 BCE
            bce_pixel = self.bce_func(preds, target_clean)
            # 只保留有效区域的 Loss，并取平均
            if valid_mask.sum() > 0:
                loss_bce = (bce_pixel * valid_mask).sum() / valid_mask.sum()
            else:
                loss_bce = torch.tensor(0.0, device=preds.device, requires_grad=True)

        # --- 2. Dice Loss (Masked) ---
        loss_dice = 0.0
        if self.dice_weight > 0:
            pred_prob = torch.sigmoid(preds)

            # 只在有效区域计算 Dice
            # 平滑项设为 1e-5 防止除零
            intersection = (pred_prob * target_clean * valid_mask).sum()
            denominator = (pred_prob * valid_mask).sum() + (target_clean * valid_mask).sum()

            loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class SimplePseudoLoss(nn.Module):
    """
    简化的伪标签损失：引入 mask 机制用于不确定性过滤。
    """

    def __init__(self, dice_weight=1.0, ce_weight=0.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        # 🌟 必须是 reduction='none' 才能和 mask 逐像素相乘
        self.bce_func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, target, mask=None):
        """
        preds: Logits
        target: 0/1 Labels (生成的伪标签)
        mask: 🌟 新增！0/1 掩码，为 1 表示可靠保留，为 0 表示不可靠丢弃
        """
        target = target.float()

        # --- 1. Dice Loss ---
        loss_dice = 0.0
        if self.dice_weight > 0:
            pred_prob = torch.sigmoid(preds)

            # 🌟 如果传入了掩码，将预测和目标都先过一遍过滤网
            if mask is not None:
                intersection = (pred_prob * target * mask).sum()
                denominator = (pred_prob * mask).sum() + (target * mask).sum()
            else:
                intersection = (pred_prob * target).sum()
                denominator = pred_prob.sum() + target.sum()

            loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)

        # --- 2. CE Loss (可选) ---
        loss_ce = 0.0
        if self.ce_weight > 0:
            ce_pixel = self.bce_func(preds, target)

            if mask is not None:
                if mask.sum() > 0:
                    loss_ce = (ce_pixel * mask).sum() / mask.sum()
                else:
                    loss_ce = torch.tensor(0.0, device=preds.device)
            else:
                loss_ce = ce_pixel.mean()

        return self.dice_weight * loss_dice + self.ce_weight * loss_ce