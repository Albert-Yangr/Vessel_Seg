import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseSliceLoss(nn.Module):
    """
    稀疏切片损失函数 (Sparse Slice Loss)
    专门用于处理 3D 图像中“只有部分 2D 切片有标注，其余全为未知”的弱监督场景。

    核心思想：
    1. 在有医生标注的切片上：使用严格的 BCE + Dice 进行强力监督。
    2. 在没有标注的空白区域：利用原图的“灰度相似性 (Affinity)”，引导血管的预测沿着相似的解剖结构向上下切片蔓延（扩散）。
    3. 全局平滑 (TV)：压制噪点，让长出来的血管保持表面平滑。
    """

    def __init__(self,
                 bce_weight=1.0,
                 dice_weight=1.0,  # Dice 负责切片内的形状约束 (只在有标签的区域算)
                 affinity_weight=10.0,  # 【核心】亲和力扩散权重，负责向垂直或未标注方向扩散伪标签 (建议较高: 10.0 - 20.0)
                 tv_weight=0.1,  # 全变分损失权重，负责全局平滑降噪
                 ignore_index=255,  # 极其关键！表示“未标注/未知区域”的标签值
                 **kwargs):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.affinity_weight = float(affinity_weight)
        self.tv_weight = float(tv_weight)
        # 255 代表这块区域医生没画，既不能当成背景0，也不能当成血管1，必须忽略
        self.ignore_index = int(ignore_index)

        if kwargs:
            print(f"⚠️ SparseSliceLoss 忽略参数: {list(kwargs.keys())}")

    def forward(self, pred_logits, slice_label, images):
        """
        前向传播计算 Loss
        Args:
            pred_logits: (B, 1, D, H, W) 模型输出的预测值 (未经过 Sigmoid 的 Logits)
            slice_label: (B, 1, D, H, W) 稀疏标签 -> 0=背景, 1=血管, 255=未知(医生没画)
            images: (B, 1, D, H, W) 原始 CT/MRI 图像，提供给 Affinity 算灰度相似度用
        """
        # ==========================================
        # 1. 强力监督 (BCE + Dice) - 只在有标注的切片上生效
        # ==========================================
        # 生成有效区域掩膜：只要标签不是 255 (即为 0 或 1)，这块区域就是有效的
        valid_mask = (slice_label != self.ignore_index).float()
        # 提取真实的前景目标 (把 1 挑出来，255 和 0 都会变成 0)
        target = (slice_label == 1).float()

        # A. 标准 BCE (逐像素计算，先不求均值，reduction='none')
        # 返回形状依然是 (B, 1, D, H, W)
        bce_pixel = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        # 如果这个 Batch 里哪怕有一丁点有效标注 (避免全空切片导致的除零报错)
        if valid_mask.sum() > 0:
            # 💡 掩码 BCE 计算：把有效区域的 loss 挑出来求和，再除以有效区域的像素总数
            loss_bce = (bce_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)

            # B. Slice Dice (只在标注了的切片上计算 Dice)
            pred_probs = torch.sigmoid(pred_logits)  # Logits -> 概率 (0~1)

            # 计算交集：预测概率 * 真实目标 * 有效掩膜 (过滤掉未标注区域的干扰)
            intersection = (pred_probs * target * valid_mask).sum()
            # 计算分母：(预测概率 + 真实目标) * 有效掩膜
            denominator = ((pred_probs + target) * valid_mask).sum()

            # 加上 1e-5 平滑项防报错
            loss_dice = 1.0 - (2.0 * intersection) / (denominator + 1e-5)
        else:
            # 极端情况：如果这张图完全没有标签 (全是 255)，强力监督 Loss 为 0
            loss_bce = torch.tensor(0.0, device=pred_logits.device)
            loss_dice = torch.tensor(0.0, device=pred_logits.device)

        # ==========================================
        # 2. 3D 亲和力扩散 Loss (Affinity Loss)
        # ==========================================
        # 作用：让未标注区域的预测结果，跟相邻切片的预测结果对齐（前提是原图长得像）。
        probs = torch.sigmoid(pred_logits)
        loss_aff = self.compute_affinity_loss(probs, images, sigma=1.0)

        # ==========================================
        # 3. 全变分损失 (TV) - 降噪
        # ==========================================
        # 作用：压制模型瞎猜产生的高频噪点，让预测出来的 3D 血管表面更加圆润光滑。
        loss_tv = self.compute_tv_loss(probs)

        # 最终将所有 Loss 按权重相加
        return (self.bce_weight * loss_bce) + \
            (self.dice_weight * loss_dice) + \
            (self.affinity_weight * loss_aff) + \
            (self.tv_weight * loss_tv)

    def compute_affinity_loss(self, probs, images, sigma=1.0):
        """
        计算 3D 亲和力损失 (Affinity Loss) - 实现“图相似则预测同”的逻辑。

        原理：
        看相邻的两个体素 (比如上下两层切片的同一个位置)，
        如果它们在原图 (images) 里的灰度值非常接近 -> 它们很可能是同一个组织 (血管或背景)。
        那么它们的预测概率 (probs) 也应该非常接近。如果原图很像但预测概率差很大，就进行严厉惩罚。
        """
        loss_sum = 0.0
        # 遍历 D(深度Z轴: 2), H(高度Y轴: 3), W(宽度X轴: 4) 三个空间维度
        for dim in [2, 3, 4]:
            # 利用 torch.narrow 进行巧妙的“错位切片”来获取相邻像素
            # img_curr: 原图的前 N-1 层； img_next: 原图的后 N-1 层 (相当于把图往旁边挪了一格)
            img_curr = torch.narrow(images, dim, 0, images.size(dim) - 1)
            img_next = torch.narrow(images, dim, 1, images.size(dim) - 1)

            # 同理，获取相邻的预测概率图
            prob_curr = torch.narrow(probs, dim, 0, probs.size(dim) - 1)
            prob_next = torch.narrow(probs, dim, 1, probs.size(dim) - 1)

            # A. 计算原图的灰度差异
            diff_img = torch.abs(img_curr - img_next)
            # B. 差异 -> 权重转换：灰度差异越小(越像)，指数 exp(-diff) 越接近 1，惩罚权重越大。
            # sigma 调节对灰度差异的敏感度。
            weight = torch.exp(-diff_img / sigma)

            # C. 计算模型预测的差异 (MSE: 平方误差)
            diff_prob = torch.pow(prob_curr - prob_next, 2)

            # 权重 * 预测差异：如果原图很像(weight大)，但模型给出了截然不同的预测(diff_prob大)，就会产生巨大的 Loss
            loss_sum += (weight * diff_prob).mean()

        return loss_sum

    def compute_tv_loss(self, probs):
        """
        计算全变分损失 (Total Variation Loss)

        原理：这是图像处理中经典的去噪算法。
        它直接惩罚相邻像素间预测概率的剧烈波动（即梯度）。
        它不管原图长什么样，纯粹要求模型的输出必须是“平缓过渡”的，不能出现斑点状的预测。
        """
        batch_size = probs.size(0)
        h_x, w_x, d_x = probs.shape[2], probs.shape[3], probs.shape[4]

        # 分别计算沿高度、宽度、深度方向的相邻像素预测差值的平方和
        # 同样利用了 Python 的切片切出偏移一格的张量相减
        h_tv = torch.pow((probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :]), 2).sum()
        w_tv = torch.pow((probs[:, :, :, 1:, :] - probs[:, :, :, :-1, :]), 2).sum()
        d_tv = torch.pow((probs[:, :, :, :, 1:] - probs[:, :, :, :, :-1]), 2).sum()

        # 获取总像素数量用于归一化
        count = batch_size * h_x * w_x * d_x

        # 乘以 2 是一种常数缩放习惯，返回这三个方向变分之和的均值
        return 2 * (h_tv + w_tv + d_tv) / count