import os
import shutil
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from pathlib import Path
from tqdm import tqdm
import warnings

# ================= 配置区域 =================
# 1. 原始数据路径
ORIGIN_ROOT = Path("/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-origin")
IMG_DIR = ORIGIN_ROOT / "test_data"
MASK_DIR = ORIGIN_ROOT / "test_mask"

# 2. 目标输出路径
TARGET_ROOT = Path("/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-S/test_resize")

# 3. 尺寸阈值 (小于此值将触发 2倍放大)
SIZE_THRESHOLD = 300


# ===========================================

def step1_resize_and_copy():
    print(f"\n🚀 [Step 1] 开始数据重采样与标准化...")

    if not IMG_DIR.exists() or not MASK_DIR.exists():
        print(f"❌ 错误: 源目录不存在。\n请检查: {IMG_DIR}\n或: {MASK_DIR}")
        return False

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    # 获取文件列表
    image_files = sorted(list(IMG_DIR.glob("*.nii.gz")))
    print(f"🔍 找到 {len(image_files)} 个原始数据文件")

    count_resized = 0
    count_copied = 0

    for img_path in tqdm(image_files, desc="处理进度"):
        file_name = img_path.name
        mask_path = MASK_DIR / file_name

        if not mask_path.exists():
            # print(f"⚠️ [跳过] 缺少 Mask: {file_name}")
            continue

        # --- ID 格式化 ---
        raw_id = file_name.replace(".nii.gz", "")
        try:
            new_id = str(int(raw_id))  # "000" -> "0"
        except ValueError:
            new_id = raw_id

        # 创建单个病例文件夹
        case_dir = TARGET_ROOT / new_id
        case_dir.mkdir(exist_ok=True)

        # 定义标准输出路径 (强制后缀为 .img.nii.gz 和 .label.nii.gz)
        target_img_path = case_dir / f"{new_id}.img.nii.gz"
        target_mask_path = case_dir / f"{new_id}.label.nii.gz"

        try:
            # 读取图像以检查尺寸
            itk_img = sitk.ReadImage(str(img_path))
            w, h, d = itk_img.GetSize()

            # --- 分支 A: 小图放大 ---
            if w < SIZE_THRESHOLD or h < SIZE_THRESHOLD:
                # 读取 Mask
                itk_mask = sitk.ReadImage(str(mask_path))

                arr_img = sitk.GetArrayFromImage(itk_img)  # (D, H, W)
                arr_mask = sitk.GetArrayFromImage(itk_mask)

                # 定义缩放因子 (Z轴不变=1, Y轴=2, X轴=2)
                zoom_factors = [1.0, 2.0, 2.0]

                # 插值
                res_img_arr = scipy.ndimage.zoom(arr_img, zoom_factors, order=3, mode='nearest')
                res_mask_arr = scipy.ndimage.zoom(arr_mask, zoom_factors, order=0, mode='nearest')

                # 转回 ITK
                new_itk_img = sitk.GetImageFromArray(res_img_arr)
                new_itk_mask = sitk.GetImageFromArray(res_mask_arr)

                # 修正元数据 (像素变多，Spacing 变小)
                orig_spacing = itk_img.GetSpacing()
                new_spacing = (orig_spacing[0] / 2.0, orig_spacing[1] / 2.0, orig_spacing[2])

                # 复制并应用新元数据
                for img_obj in [new_itk_img, new_itk_mask]:
                    img_obj.SetOrigin(itk_img.GetOrigin())
                    img_obj.SetDirection(itk_img.GetDirection())
                    img_obj.SetSpacing(new_spacing)

                # 保存 (SimpleITK 会根据后缀自动处理格式)
                sitk.WriteImage(new_itk_img, str(target_img_path))
                sitk.WriteImage(new_itk_mask, str(target_mask_path), useCompression=True)

                count_resized += 1

            # --- 分支 B: 大图直接复制 ---
            else:
                # 即使是复制，shutil.copy2 也会将其重命名为 target_img_path 指定的名字
                # 这样就避免了后缀名混乱的问题
                shutil.copy2(img_path, target_img_path)
                shutil.copy2(mask_path, target_mask_path)
                count_copied += 1

        except Exception as e:
            print(f"❌ 处理失败 {file_name}: {e}")

    print(f"✅ Step 1 完成! (放大: {count_resized}, 复制: {count_copied})")
    return True


def step2_check_and_fix_names():
    print(f"\n🔍 [Step 2] 开始文件名校验与修复...")

    if not TARGET_ROOT.exists():
        print(f"❌ 错误: 目标目录不存在 {TARGET_ROOT}")
        return

    subdirs = [d for d in TARGET_ROOT.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda x: int(x.name) if x.name.isdigit() else x.name)

    rename_count = 0
    error_count = 0

    for folder in tqdm(subdirs, desc="校验文件"):
        case_id = folder.name

        # 1. 检查并修复 .hdr.gz (遗留问题修复)
        bad_extensions = [".hdr.gz", ".nii", ".img"]  # 可能出现的非标准后缀

        for bad_ext in bad_extensions:
            # 检查 Image
            bad_img = folder / f"{case_id}{bad_ext}"
            good_img = folder / f"{case_id}.img.nii.gz"
            if bad_img.exists() and not good_img.exists():
                try:
                    bad_img.rename(good_img)
                    rename_count += 1
                except Exception as e:
                    print(f"  ❌ 重命名失败 {bad_img.name}: {e}")

            # 检查 Label
            bad_mask = folder / f"{case_id}.label{bad_ext}"  # 假设 label 也有可能错
            # 或者单纯就是 {case_id}{bad_ext} 但这和 image 冲突，这里主要针对 image 后缀

        # 2. 最终完整性检查
        final_img = folder / f"{case_id}.img.nii.gz"
        final_mask = folder / f"{case_id}.label.nii.gz"

        issues = []
        if not final_img.exists():
            issues.append("Image缺失")
        if not final_mask.exists():
            issues.append("Mask缺失")

        if issues:
            print(f"  ⚠️ Case {case_id} 异常: {', '.join(issues)}")
            error_count += 1

    print("-" * 40)
    print(f"🎉 全部流程结束!")
    print(f"  - 修复文件名数量: {rename_count}")
    print(f"  - 最终异常案例数: {error_count}")

    if error_count == 0:
        print(f"\n✨ 数据集验证完美通过！输出路径:\n{TARGET_ROOT}")
    else:
        print(f"\n⚠️ 请检查上述报错的 Case ID")


if __name__ == "__main__":
    # 忽略 scipy 警告
    warnings.filterwarnings("ignore")

    # 执行流程
    if step1_resize_and_copy():
        step2_check_and_fix_names()