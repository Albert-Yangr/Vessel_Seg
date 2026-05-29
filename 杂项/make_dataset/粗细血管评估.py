import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_opening
from skimage.morphology import skeletonize

# ==========================================
# 1. 基础配置区
# ==========================================
GT_DIR = "/home/yangrui/Project/Base-model/local_results/output/CAS2023/CAS2023-refile/GT"
PRED_DIR = "/home/yangrui/Project/Base-model/local_results/output/CAS2023/CAS2023-refile/双分支标签"

# 粗细血管分割的球形半径
VESSEL_SIZE_THRESHOLD = 3

# 🌟 新增：是否计算 clDice 开关
# 设为 False 时极速运行，只计算 Dice。设为 True 时计算 clDice (非常耗时)
CALCULATE_CLDICE = False


# ==========================================
# 2. 核心算法函数
# ==========================================
def separate_thick_thin(mask_np, radius=3):
    """
    使用形态学开运算（滚珠模型）将3D血管物理切分为“粗血管”和“细血管”
    """
    z, y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
    spherical_structure = (x ** 2 + y ** 2 + z ** 2) <= (radius ** 2)

    thick_mask = binary_opening(mask_np > 0, structure=spherical_structure)
    thin_mask = (mask_np > 0) ^ thick_mask  # 异或操作提取剥离出的细血管

    return thick_mask, thin_mask


def calculate_dice(pred, gt):
    """计算 Dice 系数"""
    pred = pred > 0
    gt = gt > 0
    intersection = np.sum(pred & gt)
    union = np.sum(pred) + np.sum(gt)

    if union == 0:
        return 1.0  # 都是背景
    return 2.0 * intersection / union


def calculate_cldice(pred, gt):
    """计算 clDice (Centerline Dice) 系数"""
    pred = pred > 0
    gt = gt > 0

    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 1.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 0.0

    # 提取 3D 骨架 (耗时瓶颈)
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)

    # 计算 Tprec 和 Tsens
    tprec = np.sum(skel_pred & gt) / (np.sum(skel_pred) + 1e-8)
    tsens = np.sum(skel_gt & pred) / (np.sum(skel_gt) + 1e-8)

    if tprec + tsens == 0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)


# ==========================================
# 3. 主评测流程
# ==========================================
def main():
    if not os.path.exists(GT_DIR):
        print(f"❌ 找不到 GT 文件夹: {GT_DIR}")
        return

    gt_files = [f for f in os.listdir(GT_DIR) if f.endswith('.nii.gz')]
    if not gt_files:
        print(f"❌ GT 文件夹为空或没有 .nii.gz 文件: {GT_DIR}")
        return

    sample_ids = []
    for f in gt_files:
        idx = f.replace('.label.nii.gz', '').replace('.nii.gz', '')
        sample_ids.append(idx)

    try:
        sample_ids.sort(key=lambda x: int(x))
    except ValueError:
        sample_ids.sort()

    print(f"🚀 开始血管分割评估任务 (共发现 {len(sample_ids)} 个样本)")
    if not CALCULATE_CLDICE:
        print(f"⚡ [极速模式] clDice 计算已关闭")

    # 根据开关动态生成表头和分隔线
    if CALCULATE_CLDICE:
        sep_len = 105
        header = f"| {'ID':<5} | {'Thick %':<8} | {'Thin %':<8} | {'Overall Dice':<12} | {'Overall clDice':<14} | {'Thick Dice':<10} | {'Thick clDice':<12} | {'Thin Dice':<9} | {'Thin clDice':<11} |"
        sep = "|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 11 + "|" + "-" * 13 + "|"
    else:
        sep_len = 69
        header = f"| {'ID':<5} | {'Thick %':<8} | {'Thin %':<8} | {'Overall Dice':<12} | {'Thick Dice':<10} | {'Thin Dice':<9} |"
        sep = "|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 12 + "|" + "-" * 11 + "|"

    print("-" * sep_len)
    print(header)
    print(sep)

    # 存储指标
    metrics = {
        'thick_ratio': [], 'thin_ratio': [],
        'overall_dice': [], 'thick_dice': [], 'thin_dice': []
    }
    if CALCULATE_CLDICE:
        metrics.update({'overall_cldice': [], 'thick_cldice': [], 'thin_cldice': []})

    for idx in sample_ids:
        gt_path = os.path.join(GT_DIR, f"{idx}.label.nii.gz")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(GT_DIR, f"{idx}.nii.gz")

        pred_path = os.path.join(PRED_DIR, f"{idx}_pred.nii.gz")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            missing_msg = '[预测文件缺失]'
            if CALCULATE_CLDICE:
                print(f"| {idx:<5} | {missing_msg:<90} |")
            else:
                print(f"| {idx:<5} | {missing_msg:<54} |")
            continue

        gt_np = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(bool)
        pred_np = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).astype(bool)

        # 分离粗细血管
        gt_thick, gt_thin = separate_thick_thin(gt_np, VESSEL_SIZE_THRESHOLD)
        pred_thick, pred_thin = separate_thick_thin(pred_np, VESSEL_SIZE_THRESHOLD)

        total_gt_voxels = np.sum(gt_np)
        if total_gt_voxels > 0:
            thick_ratio = np.sum(gt_thick) / total_gt_voxels
            thin_ratio = np.sum(gt_thin) / total_gt_voxels
        else:
            thick_ratio = thin_ratio = 0.0

        metrics['thick_ratio'].append(thick_ratio)
        metrics['thin_ratio'].append(thin_ratio)

        # 计算 Dice
        o_dice = calculate_dice(pred_np, gt_np)
        thick_dice = calculate_dice(pred_thick, gt_thick)
        thin_dice = calculate_dice(pred_thin, gt_thin)

        metrics['overall_dice'].append(o_dice)
        metrics['thick_dice'].append(thick_dice)
        metrics['thin_dice'].append(thin_dice)

        # 条件计算 clDice 并打印单行报表
        if CALCULATE_CLDICE:
            o_cldice = calculate_cldice(pred_np, gt_np)
            thick_cldice = calculate_cldice(pred_thick, gt_thick)
            thin_cldice = calculate_cldice(pred_thin, gt_thin)

            metrics['overall_cldice'].append(o_cldice)
            metrics['thick_cldice'].append(thick_cldice)
            metrics['thin_cldice'].append(thin_cldice)

            print(
                f"| {idx:<5} | {thick_ratio * 100:>7.2f}% | {thin_ratio * 100:>7.2f}% | {o_dice * 100:>9.2f}%   | {o_cldice * 100:>11.2f}%   | {thick_dice * 100:>7.2f}%   | {thick_cldice * 100:>9.2f}%   | {thin_dice * 100:>6.2f}%   | {thin_cldice * 100:>8.2f}%   |")
        else:
            print(
                f"| {idx:<5} | {thick_ratio * 100:>7.2f}% | {thin_ratio * 100:>7.2f}% | {o_dice * 100:>9.2f}%   | {thick_dice * 100:>7.2f}%   | {thin_dice * 100:>6.2f}%   |")

    print("-" * sep_len)

    # 计算并打印平均值
    if metrics['overall_dice']:
        avg_thick_ratio = np.mean(metrics['thick_ratio']) * 100
        avg_thin_ratio = np.mean(metrics['thin_ratio']) * 100
        avg_o_dice = np.mean(metrics['overall_dice']) * 100
        avg_thick_dice = np.mean(metrics['thick_dice']) * 100
        avg_thin_dice = np.mean(metrics['thin_dice']) * 100

        if CALCULATE_CLDICE:
            avg_o_cldice = np.mean(metrics['overall_cldice']) * 100
            avg_thick_cldice = np.mean(metrics['thick_cldice']) * 100
            avg_thin_cldice = np.mean(metrics['thin_cldice']) * 100
            print(
                f"| {'AVG':<5} | {avg_thick_ratio:>7.2f}% | {avg_thin_ratio:>7.2f}% | {avg_o_dice:>9.2f}%   | {avg_o_cldice:>11.2f}%   | {avg_thick_dice:>7.2f}%   | {avg_thick_cldice:>9.2f}%   | {avg_thin_dice:>6.2f}%   | {avg_thin_cldice:>8.2f}%   |")
        else:
            print(
                f"| {'AVG':<5} | {avg_thick_ratio:>7.2f}% | {avg_thin_ratio:>7.2f}% | {avg_o_dice:>9.2f}%   | {avg_thick_dice:>7.2f}%   | {avg_thin_dice:>6.2f}%   |")

    print("=" * sep_len)


if __name__ == "__main__":
    main()