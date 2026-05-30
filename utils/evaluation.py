from pathlib import Path

import numpy as np
import torch
# skimage 用于图像处理，这里主要用于提取形态学骨架（中心线）和计算拓扑不变量（欧拉数、连通域）
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.measure import euler_number, label
# sklearn 用于计算传统的二分类指标
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import SimpleITK as sitk

# [新增] 导入 MONAI 指标库，用于计算 3D 表面距离相关的高阶医学指标
from monai.metrics import compute_average_surface_distance, compute_surface_dice


class Evaluator:
    """
    血管分割评估器类 (Vessel Segmentation Evaluator)

    实现了一系列专门针对管状/细微结构的评估指标，包括：
    - Dice系数 (Dice Score)：评估整体体积的重叠度。
    - clDice (Centerline Dice)：基于中心线（骨架）的Dice，专门评估管状结构的拓扑连通性。
    - Betti数误差 (Betti Number Error)：纯数学拓扑特征（如连通分支数、孔洞数）的误差评估。
    - NSD (Normalized Surface Dice)：归一化表面Dice，评估预测边界与真实边界在一定容差内的重合度。
    - ASD (Average Surface Distance)：平均表面距离，评估预测边界到真实边界的平均物理偏差。
    - 传统分类指标：精确度(Precision)、召回率/敏感度(Recall/Sensitivity)、特异度(Specificity)等。
    """

    def extract_labels(self, gt_array, pred_array):
        """
        提取真实值(GT)和预测值(Pred)中包含的所有唯一标签（类别）。
        由于我们是二分类（血管/背景），正常返回的应该是 [0, 1]。
        """
        labels_gt = np.unique(gt_array)
        labels_pred = np.unique(pred_array)
        # 取并集，确保无论哪边漏了某个类别，都能被收集到
        labels = list(set().union(labels_gt, labels_pred))
        labels = [int(x) for x in labels]
        return labels

    def betti_number_error(self, gt, pred):
        """
        计算 Betti 数误差 - 用于评估拓扑结构差异。
        Betti数是代数拓扑中的概念。在3D空间中：
        - Betti-0 (b0): 表示连通分支的数量（有多少根独立的血管）。
        - Betti-1 (b1): 表示一维孔洞/环的数量（血管形成的闭环结构）。
        - Betti-2 (b2): 表示二维空腔的数量（如气球内部的空腔，血管通常没有）。
        """
        labels = self.extract_labels(gt_array=gt, pred_array=pred)
        # 移除背景标签(0)，只计算前景(血管)的拓扑特征
        if 0 in labels:
            labels.remove(0)

        # 如果连前景都没有，说明全黑，误差为0
        if len(labels) == 0:
            return 0, 0

        # 分别计算真实标签和预测结果的 Betti 数 [b0, b1, b2]
        gt_betti_numbers = self.betti_number(gt)
        pred_betti_numbers = self.betti_number(pred)

        # 计算 b0 (连通分支数) 和 b1 (环数) 的绝对误差
        betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])

        return betti_0_error, betti_1_error

    def betti_number(self, img):
        """
        利用欧拉示性数 (Euler Characteristic) 快速计算 3D 二值图像的 Betti 数。
        数学公式: 欧拉数 (Euler) = b0 - b1 + b2
        """
        assert img.ndim == 3  # 确保是3D数据
        # 定义 3D 空间中的邻域连通性：
        N6 = 1  # 6连通：只算上下左右前后 6个面相邻
        N26 = 3  # 26连通：算上面+棱+顶点，共 26个相邻点

        # 边缘填充，防止边界上的血管计算拓扑时出错
        padded = np.pad(img, pad_width=1)

        # 计算 b0 (连通分支数)：使用 26 连通寻找前景(1)的连通域数量
        _, b0 = label(padded, return_num=True, connectivity=N26)

        # 计算欧拉数：skimage 提供的计算整体欧拉示性数的函数
        euler_char_num = euler_number(padded, connectivity=N26)

        # 计算 b2 (空腔数)：空腔其实就是背景(0)中，完全被前景包裹的 6连通 连通域数量
        _, b2 = label(1 - padded, return_num=True, connectivity=N6)
        b2 -= 1  # 减去外部无限大背景的那个连通域

        # 根据欧拉公式倒推 b1 (环数): b1 = b0 + b2 - Euler
        b1 = b0 + b2 - euler_char_num
        return [b0, b1, b2]

    def cl_dice(self, v_p, v_l):
        """
        计算拓扑感知的 clDice (Centerline Dice) 系数。
        原理：它不看你预测的血管有多粗，而是看你预测的【血管体积】是否包住了真实的【中心线】，
        以及真实的【血管体积】是否包住了你预测的【中心线】。这能极大地惩罚血管断裂现象。
        """

        def cl_score(v, s):
            # v: volume(体积掩码), s: skeleton(骨架/中心线掩码)
            # 计算骨架落在体积内部的比例
            return np.sum(v * s) / np.sum(s)

        # 根据维度选择对应的 2D 或 3D 骨架提取算法
        if len(v_p.shape) == 2:
            # tprec (拓扑精确率): 真实的骨架 有多少落在了 预测的体积内
            tprec = cl_score(v_p, skeletonize(v_l))
            # tsens (拓扑召回率): 预测的骨架 有多少落在了 真实的体积内
            tsens = cl_score(v_l, skeletonize(v_p))
        elif len(v_p.shape) == 3:
            tprec = cl_score(v_p, skeletonize_3d(v_l))
            tsens = cl_score(v_l, skeletonize_3d(v_p))
        else:
            raise ValueError(f"Invalid shape for cl_dice: {v_p.shape}")

        # 使用 F1-score 的调和平均公式计算最终的 clDice
        # 加上 np.finfo(float).eps (极小值) 防止分母为 0 导致 NaN
        return 2 * tprec * tsens / (tprec + tsens + np.finfo(float).eps)

    def estimate_metrics(self, pred_seg, gt_seg, threshold=0.5, spacing=(1, 1, 1), nsd_tolerance=1.0, fast=False):
        """
        核心评估函数：计算全面的分割评估指标 (包含像素级、距离级、拓扑级)。

        Args:
            pred_seg: 模型预测的分割概率图 (Tensor，通常经过 Sigmoid，值在 0~1 之间)
            gt_seg: 真实分割金标准 (Tensor，0 或 1)
            threshold: 概率二值化阈值，默认 0.5
            spacing: 图像的物理间距 (z, y, x)，用于将像素距离转化为真实的毫米(mm)距离
            nsd_tolerance: NSD 的容差阈值 (默认 1.0mm 范围内算对)
            fast: 快速模式标志 (验证集可用，只算Dice，节约时间)

        Returns:
            metrics: 包含所有评估指标结果的字典
        """
        metrics = {}

        # 1. 基础数据准备
        # 将概率图按照 threshold 二值化，并从 GPU 转移到 CPU 上，因为后续的 Numpy/Scipy 算法只能在 CPU 跑
        pred_seg_thresh = (pred_seg >= threshold).float().cpu()
        gt_seg_cpu = gt_seg.cpu()

        # 2. 计算混淆矩阵 (Confusion Matrix)
        # 将 3D 矩阵展平为 1D 向量 (flatten)，计算真负(TN)、假正(FP)、假负(FN)、真正(TP)像素点数
        tn, fp, fn, tp = confusion_matrix(
            gt_seg_cpu.flatten().clone().numpy(),
            pred_seg_thresh.flatten().clone().numpy(),
            labels=[0, 1],
        ).ravel()

        # --- 快速模式返回 ---
        # 如果开启了 fast=True，为了加快验证阶段的速度，只计算最核心的 Dice 就返回
        if fast:
            metrics["dice"] = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            return metrics

        # =======================================================
        # 3. 计算基于边界距离的指标: ASD 和 NSD (使用 MONAI 库)
        # =======================================================
        # 深度拷贝张量，防止后续维度变换影响原数据
        pred_monai = pred_seg_thresh.clone()
        gt_monai = gt_seg_cpu.clone()

        # MONAI 的距离计算函数严格要求输入格式必须是 5D 张量: (Batch, Channel, Spatial_Z, Spatial_Y, Spatial_X)
        # 循环补充在前面缺失的 Batch 和 Channel 维度 (unsqueeze(0))
        while pred_monai.ndim < 5:
            pred_monai = pred_monai.unsqueeze(0)
        while gt_monai.ndim < 5:
            gt_monai = gt_monai.unsqueeze(0)

        # [新增] 计算 Average Surface Distance (ASD - 平均表面距离)
        # symmetric=True 表示计算双向平均距离 (预测到真实的距离 + 真实到预测的距离) / 2
        # spacing 确保算出来的是物理距离 (mm) 而不是单纯的像素个数
        try:
            asd = compute_average_surface_distance(
                y_pred=pred_monai,
                y=gt_monai,
                symmetric=True,
                spacing=spacing
            )
            # 处理 MONAI 返回 tensor 的情况，转为 Python 标量
            metrics["asd"] = asd.item() if isinstance(asd, torch.Tensor) else asd
            # 如果预测图全是黑的，距离可能趋于无穷大 (inf) 或无意义 (nan)，兜底设为 0.0 或惩罚值
            if np.isinf(metrics["asd"]) or np.isnan(metrics["asd"]):
                metrics["asd"] = 0.0
        except Exception as e:
            # 捕获空预测或全满预测导致的报错
            metrics["asd"] = 0.0

        # [新增] 计算 Normalized Surface Dice (NSD - 归一化表面Dice)
        # 它表示：预测边界上，有多少比例的点落在了真实边界的容差 (nsd_tolerance) 范围内。
        # 这是目前医学竞赛中非常看重的边界评估指标。
        try:
            nsd = compute_surface_dice(
                y_pred=pred_monai,
                y=gt_monai,
                class_thresholds=[nsd_tolerance],  # 容忍距离 (mm)
                spacing=spacing
            )
            metrics["nsd"] = nsd.item() if isinstance(nsd, torch.Tensor) else nsd
            # NSD 的理论值域是 0.0 到 1.0
        except Exception as e:
            metrics["nsd"] = 0.0

        # =======================================================
        # 4. 计算其他复杂指标 (基于 numpy 和 sklearn)
        # =======================================================
        # 分别取二值化的 Numpy 数组 (用于拓扑) 和 概率/原始格式 (用于 AUC)
        gt_np = gt_seg_cpu.flatten().clone().detach().numpy()
        pred_np = pred_seg.flatten().cpu().clone().detach().numpy()
        pred_thresh_np = pred_seg_thresh.flatten().clone().detach().numpy()

        # 计算 ROC 曲线下的面积 (ROC AUC)
        try:
            roc_auc = roc_auc_score(gt_np, pred_np)
        except ValueError:
            # 异常处理：如果当前 Batch 里面全都是背景（没有正样本），AUC 无法计算
            roc_auc = 0.0

        # 计算 PR 曲线下的面积 (PR AUC，对类别不平衡的血管分割更有参考价值)
        try:
            pr_auc = average_precision_score(gt_np, pred_np)
        except ValueError:
            pr_auc = 0.0

        # 计算 clDice 和 拓扑指标 (去除多余的空维度，转为纯 3D numpy 数组，且类型转为 uint8(byte))
        pred_3d = pred_seg_thresh.squeeze().clone().detach().byte().numpy()
        gt_3d = gt_seg_cpu.squeeze().clone().detach().byte().numpy()

        # 调用中心线 Dice 计算
        cldice = self.cl_dice(pred_3d, gt_3d)

        # 计算 Betti 误差
        # 注意：astype(int) 确保图像数据类型在拓扑算法中不溢出
        betti_0_error, betti_1_error = self.betti_number_error(
            gt_3d.astype(int),
            pred_3d.astype(int)
        )

        # 记录模型单方面预测出的 Betti 数
        betti_0, betti_1, betti_2 = self.betti_number(pred_3d.astype(int))

        # =======================================================
        # 5. 汇总所有传统的体素级分类指标
        # =======================================================
        epsilon = 1e-6  # 极小值，防止由于除以 0 导致程序崩溃

        # 召回率 (Recall) / 敏感度 (Sensitivity) / TPR: 金标准中，被正确预测为血管的比例
        metrics["recall_tpr_sensitivity"] = tp / (tp + fn + epsilon)

        # 假阳性率 (FPR): 背景中被错误预测为血管的比例
        metrics["fpr"] = fp / (fp + tn + epsilon)

        # 精确率 (Precision): 预测为血管的像素中，真正是血管的比例
        metrics["precision"] = tp / (tp + fp + epsilon)

        # 特异度 (Specificity): 背景中被正确预测为背景的比例
        metrics["specificity"] = tn / (tn + fp + epsilon)

        # 交并比 (Jaccard / IoU): TP / (TP + FP + FN)
        metrics["jaccard_iou"] = tp / (tp + fp + fn + epsilon)

        # Dice 系数 (F1-score): 2 * TP / (2*TP + FP + FN)
        metrics["dice"] = (2 * tp) / (2 * tp + fp + fn + epsilon)

        # 录入之前算好的拓扑和边界指标
        metrics["cldice"] = cldice

        # 总体准确率 (Accuracy): (正确预测正 + 正确预测负) / 所有像素点 (通常在医学中偏高，因为背景庞大)
        metrics["accuracy"] = (tp + tn) / (tn + fp + tp + fn + epsilon)

        metrics["roc_auc"] = roc_auc
        metrics["pr_auc_ap"] = pr_auc
        metrics["betti_0_error"] = betti_0_error
        metrics["betti_1_error"] = betti_1_error
        metrics["betti_0"] = betti_0
        metrics["betti_1"] = betti_1
        metrics["betti_2"] = betti_2

        return metrics


def read_nifti(path: str):
    """
    轻量级辅助工具：使用 SimpleITK 读取 NIfTI (.nii / .nii.gz) 格式的医学图像文件，
    并直接转化为 Numpy 数组。
    注意：SimpleITK 读出的数组维度顺序是 (Z, Y, X)。
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def calculate_mean_metrics(results, round_to=2):
    """
    计算整个验证集/测试集上，所有样本评估指标的平均值。

    Args:
        results: 包含多个字典的列表，每个字典是一个样本的评估结果 [ {dice:0.8...}, {dice:0.9...} ]
        round_to: 结果保留的小数位数

    Returns:
        mean: 包含各项平均指标的新字典
    """
    if not results:
        return {}

    mean = {}
    # 遍历某一个样本的字典 keys（如 'dice', 'asd', 'cldice' 等）
    for k in results[0].keys():
        # 收集所有样本在当前指标 (key) 下的值
        numbers = [r[k] for r in results]

        # 过滤掉异常值 (NaN 和 Inf)，防止污染平均分
        numbers = [n for n in numbers if not np.isnan(n) and not np.isinf(n)]

        # 如果过滤完没数据了，兜底设为 0.0
        if not numbers:
            mean[k] = 0.0
            continue

        # 计算算术平均值
        mean[k] = np.mean(numbers)

        # 【重点排版】将主要关注的 0~1 的指标转化为百分制 (0~100) 方便论文展示
        if k in ["dice", "cldice", "nsd", "jaccard_iou", "recall_tpr_sensitivity", "precision", "accuracy"]:
            mean[k] = mean[k] * 100

        # 保留指定的小数位数
        mean[k] = np.round(mean[k], round_to)

    return mean