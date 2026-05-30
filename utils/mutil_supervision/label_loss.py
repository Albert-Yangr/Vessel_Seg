import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 完整的全监督联合损失函数 (Full Label Loss)
# ==============================================================================

class FullLabelLoss(nn.Module):
    """
    联合损失函数：将 BCE Loss、Dice Loss 和 clDice Loss 结合在一起。
    BCE 负责像素级别的绝对准确率；
    Dice 负责缓解医学图像中严重的类别不平衡（背景远大于血管）；
    clDice 负责惩罚血管断裂，保持拓扑连通性。
    """
    def __init__(self, dice_weight=0.9, ce_weight=0.1, cldice_weight=0.0, **kwargs):
        super().__init__()
        # 记录各项损失的权重
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.cldice_weight = float(cldice_weight)

        # 🌟 如果配置了 cldice_weight 大于 0，说明我们要开启拓扑约束
        # 这里初始化 clDiceLoss 算子，iters=5 表示骨架提取的迭代次数（通常管径越粗，需要迭代腐蚀的次数越多）
        if self.cldice_weight > 0:
            self.cldice_fn = clDiceLoss(iters=5)

        # 打印被忽略的参数，方便调试 (防止在 Hydra 配置文件中写错了参数名但代码没报错)
        if kwargs:
            print(f"⚠️ FullLabelLoss 忽略额外参数: {list(kwargs.keys())}")

    def forward(self, pred_logits, target, images=None):
        """
        前向传播计算 Loss
        Args:
            pred_logits: (B, 1, D, H, W) 模型直接输出的预测值（Logits，未经过 Sigmoid）
            target: (B, 1, D, H, W) 真实标签 0或1
            images: 原始图像数据。在这里只是个占位符，为了保持与其他 Loss (如需要原图算亲和力的 SliceLoss) 的接口一致
        """
        # 1. 维度对齐与类型转换
        # 如果传入的 target 是 (B, D, H, W)，缺少通道维度，则手动在第 1 维扩展出通道维度
        if target.ndim == 4:
            target = target.unsqueeze(1)

        # 确保真实标签是浮点型，才能参与梯度计算
        target = target.float()

        # 2. 计算 Binary Cross Entropy (BCE) 交叉熵损失
        # 注意：这里直接使用 with_logits 版本，它在内部自动做 Sigmoid 然后算 BCE，
        # 这比先算 Sigmoid 再算 BCE 在数值上更稳定，不容易出现梯度爆炸或消失。
        loss_ce = F.binary_cross_entropy_with_logits(pred_logits, target)

        # 3. 计算 Dice Loss (交并比损失)
        # 计算 Dice 需要真正的概率值 (0~1)，所以这里手动将 Logits 经过 Sigmoid 映射
        probs = torch.sigmoid(pred_logits)

        # 计算交集 (Intersection): 预测概率图与真实标签逐像素相乘求和
        intersection = (probs * target).sum()
        # 计算并集相关的分母 (Union): 预测概率之和 + 真实标签之和
        union = (probs + target).sum()

        # Dice 公式: 1 - 2*交集 / 分母。
        # 加上 1e-5 (极小值) 是为了平滑，防止在图像全黑(无血管)时出现分母为 0 导致 NaN 的错误。
        loss_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)

        # 4. 加权求和基础损失 (Dice + BCE)
        total_loss = (self.dice_weight * loss_dice) + (self.ce_weight * loss_ce)

        # 5. 🔥 动态挂载 clDice 拓扑损失
        if self.cldice_weight > 0:
            # 注意: cldice_fn 接收的是 probs (0~1的概率) 和 target
            loss_cldice = self.cldice_fn(probs, target)
            # 叠加 clDice 损失
            total_loss += (self.cldice_weight * loss_cldice)

        return total_loss


# ==============================================================================
# 拓扑骨架提取与 clDice 损失函数 (可微形态学操作)
# ==============================================================================
# 传统的数学形态学（腐蚀、膨胀）是不可微的，无法用于深度学习反向传播。
# 这里的神仙操作是：利用 Max Pooling (最大池化) 近似实现可微的形态学操作！

def soft_erode(img):
    """
    基于池化的可微软腐蚀 (Soft Erosion)
    腐蚀的作用是让物体“变瘦”。在数学上，腐蚀等价于在局部窗口取“最小值”。
    PyTorch 没有原生的 3D 最小池化 (Min Pooling)，但我们可以用 -MaxPool(-x) 来等效实现！

    为了更好地保护 3D 空间中的细长管状结构（防止被轻易腐蚀断掉），
    这里没有用一个巨大的 3x3x3 核，而是分别沿 X、Y、Z 三个轴做 1D 的腐蚀，最后取三者的最小值。
    """
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))  # 沿深度/Z轴腐蚀
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))  # 沿高度/Y轴腐蚀
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))  # 沿宽度/X轴腐蚀
    # 取三个方向腐蚀结果的最小值
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    """
    基于池化的可微软膨胀 (Soft Dilation)
    膨胀的作用是让物体“变胖”。在数学上，膨胀等价于在局部窗口取“最大值”。
    直接使用 3x3x3 的最大池化 (Max Pooling) 完美实现，stride=1, padding=1 保证尺寸不变。
    """
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    """
    可微开运算 (Soft Opening)
    开运算 = 先腐蚀，后膨胀。
    作用：可以抹除孤立的小噪点、细小毛刺，而大块物体的整体体积基本保持不变。
    """
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iters=5):
    """
    可微动态软骨架提取 (Soft Skeletonization)
    提取 3D 物体的中心线 (骨架)。这是一种基于形态学的迭代剥离算法。
    """
    # 第一次开运算
    img1 = soft_open(img)
    # img - img1 等价于顶帽变换(Top-hat)，能提取出极其细微的、被开运算抹掉的细长结构，将其作为骨架的一部分
    skel = F.relu(img - img1)

    # 迭代循环，一层层剥去外皮
    for j in range(iters):
        img = soft_erode(img)  # 将物体剥去一层皮 (腐蚀)
        img1 = soft_open(img)  # 对剥皮后的物体做开运算
        delta = F.relu(img - img1)  # 提取出当前这一层中细长的部分

        # 将新提取出的细长部分累加到总骨架中，并确保值不会超过 1 (delta - skel*delta)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class clDiceLoss(nn.Module):
    """
    中心线 Dice 损失 (Centerline Dice Loss)
    专门针对管状/网状结构（如血管、神经、道路网）分割设计的损失函数。
    它不关心预测的血管有多粗，只关心预测出的骨架是否连通、是否吻合。
    """

    def __init__(self, iters=5, smooth=1e-5):
        super().__init__()
        self.iters = iters  # 骨架提取的腐蚀剥离次数
        self.smooth = smooth  # 平滑项，防止除零

    def forward(self, y_pred, y_true):
        # 1. 分别提取模型预测概率图和真实标签的 3D 骨架 (中心线)
        skel_pred = soft_skeletonize(y_pred, self.iters)
        skel_true = soft_skeletonize(y_true, self.iters)

        # 2. 计算拓扑精确率 (Topology Precision - Tprec)
        # 含义：模型预测出来的骨架 (skel_pred)，有多少落在了真实的血管体积 (y_true) 内部？
        # 分子：预测骨架与真实血管的交集；分母：预测骨架的总体积
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)

        # 3. 计算拓扑召回率 (Topology Sensitivity/Recall - Tsens)
        # 含义：真实的血管骨架 (skel_true)，有多少落在了模型预测的血管体积 (y_pred) 内部？
        # 分子：真实骨架与预测血管的交集；分母：真实骨架的总体积
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)

        # 4. 计算 clDice
        # 像 F1-score 一样，求精确率和召回率的调和平均数
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)

        # 返回 Loss 值 (我们希望 cl_dice 越大越好，即 1 - cl_dice 越小越好)
        return 1.0 - cl_dice