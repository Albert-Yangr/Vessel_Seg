import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接复用你之前写好的、经过验证的软骨架提取算法
from utils.TASA.cldice_loss import soft_skeletonize


class clCELoss(nn.Module):
    """
    MICCAI 2024: Centerline-Cross Entropy Loss (专门针对单通道二分类优化)
    Better Topology Consistency Without Sacrificing Accuracy
    """

    def __init__(self, iters=5):
        super().__init__()
        self.iters = iters

    def forward(self, y_pred_logits, y_true):
        # 注意：这里接收的是 logits（未经过 sigmoid 的原始输出），为了保证 CE 计算的数值稳定性

        # 1. 计算未经过 Reduction (求均值) 的逐像素二元交叉熵损失
        # l_ce 形状为 [B, 1, D, H, W]
        l_ce = F.binary_cross_entropy_with_logits(y_pred_logits, y_true.float(), reduction='none')

        # 2. 将预测值转化为概率图，用于提取预测的软骨架
        y_pred_prob = torch.sigmoid(y_pred_logits)

        # 3. 提取预测概率图和真实标签的软骨架 (中心线)
        skel_pred = soft_skeletonize(y_pred_prob, self.iters)
        skel_true = soft_skeletonize(y_true.float(), self.iters)

        # 4. 神来之笔：用骨架作为 Mask，去过滤交叉熵损失！
        # 拓扑精确率惩罚: 只在真实骨架上的像素点，计算预测的交叉熵损失
        ce_tprec = torch.mul(l_ce, skel_true).mean()

        # 拓扑召回率惩罚: 只在预测出的骨架像素点上，计算交叉熵损失 (惩罚假阳性分支)
        ce_trecall = torch.mul(l_ce, skel_pred).mean()

        # 将两股拓扑惩罚力量相加
        cl_ce = ce_tprec + ce_trecall
        return cl_ce