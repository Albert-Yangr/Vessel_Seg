import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_erode(img):
    """基于池化的软腐蚀 (Soft Erosion)"""
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)

def soft_dilate(img):
    """基于池化的软膨胀 (Soft Dilation)"""
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img, iters=5):
    """动态软骨架提取 (提取 3D 中心线)"""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

class clDiceLoss(nn.Module):
    """中心线 Dice 损失 (Centerline Dice)"""
    def __init__(self, iters=5, smooth=1e-5):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # 将概率图和标签转化为骨架
        skel_pred = soft_skeletonize(y_pred, self.iters)
        skel_true = soft_skeletonize(y_true, self.iters)

        # 召回率 (预测的体积是否包住了真实的骨架)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        # 精确率 (预测的骨架是否在真实的体积内)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)

        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
        return 1.0 - cl_dice