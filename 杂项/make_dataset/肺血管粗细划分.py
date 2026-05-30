import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label
import os

# ==========================================
# 1. 配置路径
# ==========================================
input_path = "/home/yangrui/Project/Base-model/datasets/Parse/Parse-reshape/train/317/317.slice.nii.gz"
output_path = "/home/yangrui/Project/Base-model/datasets/Parse/Parse-reshape/train/317/317.slice2.nii.gz"

# 🌟 核心参数：连通域像素面积阈值
# 决定了在 2D 切片上，截面积大于多少个像素才算“主干血管 (2)”。
# 建议值：50 ~ 200 之间。你可以根据生成的图像动态微调。
AREA_THRESHOLD = 800


def process_2d_plane(slice_2d, plane_name):
    """
    对单个 2D 切片进行连通域面积判定
    """
    # 提取血管区域 (值为 1)
    vessel_mask = (slice_2d == 1)

    # 🌟 核心算法：寻找所有独立的连通块
    # labeled_array 会给每个独立的血管块打上不同的数字标签 (1, 2, 3...)
    # num_features 是这切片里一共找到了多少块血管
    labeled_array, num_features = label(vessel_mask)

    new_slice = slice_2d.copy()
    trunk_count = 0
    branch_count = 0

    # 遍历每一个找到的血管块
    for i in range(1, num_features + 1):
        comp_mask = (labeled_array == i)
        area = comp_mask.sum()  # 计算这个血管块的像素面积

        # 如果面积大于阈值，判定为大块主干血管 (标记为 2)
        if area > AREA_THRESHOLD:
            new_slice[comp_mask] = 2
            trunk_count += 1
        else:
            branch_count += 1

    print(f"   [{plane_name}] 找到 {trunk_count} 块主干 (2), {branch_count} 块细支 (1)")
    return new_slice


def process_cross_slices():
    print(f"📥 正在读取十字切片: {input_path}")
    if not os.path.exists(input_path):
        print("❌ 文件不存在，请检查路径！")
        return

    itk_img = sitk.ReadImage(input_path)
    mask = sitk.GetArrayFromImage(itk_img)  # 形状为 (Z, Y, X)

    # 🌟 精准获取你制作切片时的十字中心坐标
    cz = mask.shape[0] // 2
    cy = mask.shape[1] // 2
    cx = mask.shape[2] // 2

    new_mask = mask.copy()

    # 🌟 针对三个切面，分别进行独立的 2D 面积判定！
    print("✂️ 开始处理 Axial (XY) 横断面...")
    new_mask[cz, :, :] = process_2d_plane(mask[cz, :, :], "XY")

    print("✂️ 开始处理 Coronal (XZ) 冠状面...")
    new_mask[:, cy, :] = process_2d_plane(mask[:, cy, :], "XZ")

    print("✂️ 开始处理 Sagittal (YZ) 矢状面...")
    new_mask[:, :, cx] = process_2d_plane(mask[:, :, cx], "YZ")

    # 保存结果
    new_itk = sitk.GetImageFromArray(new_mask)
    new_itk.CopyInformation(itk_img)  # 保留物理坐标系
    sitk.WriteImage(new_itk, output_path)

    print(f"\n✅ 处理完成，智能区分标签已保存至: {output_path}")
    print(f"👉 使用的面积阈值: {AREA_THRESHOLD} 像素")


if __name__ == "__main__":
    process_cross_slices()