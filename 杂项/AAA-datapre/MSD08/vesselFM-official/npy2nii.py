import os
import json
import glob
import shutil
import numpy as np
import SimpleITK as sitk

# ================= 核心配置区域 =================
# 输入：预处理后的数据目录 (包含 imagesTr, labelsTr)
INPUT_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-Official"

# 输出：最终生成的 NIfTI 数据集目录
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-Official-Final"

# 划分配置 (Paper: 3 Train, 1 Val, Rest Test)
TRAIN_NUM = 3
VAL_NUM = 1


# ================= 辅助函数 =================

def clean_and_create_dir(path):
    """暴力清理并重建目录，防止旧文件干扰"""
    if os.path.exists(path):
        print(f"[System] Removing old directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path)
    print(f"[System] Created clean directory: {path}")


def build_volume(slice_folder):
    """从切片文件夹重建 3D Volume"""
    # 1. 读取所有切片并排序
    slice_files = sorted(glob.glob(os.path.join(slice_folder, "*.npy")))
    if not slice_files:
        return None, None

    # 2. 堆叠数组 (Z, H, W)
    slices = [np.load(s) for s in slice_files]
    vol_array = np.stack(slices, axis=0)

    # 3. 读取元数据
    meta = {}
    meta_path = os.path.join(slice_folder, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    return vol_array, meta


def array_to_sitk(array, meta):
    """将 Numpy 数组转回 SimpleITK 对象并恢复空间信息"""
    img = sitk.GetImageFromArray(array)

    # 恢复空间元数据
    if "spacing" in meta: img.SetSpacing(meta["spacing"])
    if "origin" in meta:  img.SetOrigin(meta["origin"])
    if "direction" in meta: img.SetDirection(meta["direction"])

    return img


def save_nifti_safe(image_obj, final_path):
    """
    保存 NIfTI 的安全版本。
    逻辑：
    1. ITK 有时对 'name.img.nii.gz' 这种命名会产生误判（以为是 Analyze 格式）。
    2. 所以我们先保存为 'name_img.nii.gz' (临时文件)。
    3. 保存成功后，立即将其重命名为用户想要的 'name.img.nii.gz'。
    """
    dirname = os.path.dirname(final_path)
    filename = os.path.basename(final_path)

    # 1. 创建一个临时的安全文件名 (把点换成下划线)
    temp_filename = "TEMP_" + filename.replace(".", "_") + ".nii.gz"
    temp_path = os.path.join(dirname, temp_filename)

    try:
        # 2. 写入临时文件
        writer = sitk.ImageFileWriter()
        writer.SetFileName(temp_path)
        writer.SetImageIO("NiftiImageIO")
        writer.Execute(image_obj)

        # 3. 重命名为最终目标文件名
        if os.path.exists(final_path):
            os.remove(final_path)  # 如果目标已存在，先删除

        os.rename(temp_path, final_path)
        # print(f"      Saved: {filename}") # 调试用，太啰嗦可以注释掉

    except Exception as e:
        print(f"      [Error] Failed to save {filename}: {e}")
        # 清理残余
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ================= 主程序 =================

def main():
    # 1. 清理环境
    clean_and_create_dir(OUTPUT_DIR)

    images_in = os.path.join(INPUT_DIR, "imagesTr")
    labels_in = os.path.join(INPUT_DIR, "labelsTr")

    # 2. 获取并排序所有样本
    # 确保只读取文件夹
    all_samples = sorted([
        d for d in os.listdir(images_in)
        if os.path.isdir(os.path.join(images_in, d))
    ])

    print(f"[Info] Found {len(all_samples)} samples in total.")

    # 3. 划分数据集
    splits = {
        "fine-tuning": all_samples[:TRAIN_NUM],
        "val": all_samples[TRAIN_NUM: TRAIN_NUM + VAL_NUM],
        "test": all_samples[TRAIN_NUM + VAL_NUM:]
    }

    # 4. 开始处理
    for split_type, sample_list in splits.items():
        print(f"\n>>> Processing Split: {split_type.upper()} ({len(sample_list)} samples)")
        if len(sample_list) == 0:
            continue

        for sample_name in sample_list:
            # 提取纯数字ID (例如: "liver_10" -> "10")
            raw_id = sample_name.split('_')[-1]
            # 过滤非数字字符，防止文件名混乱
            clean_id = ''.join(filter(str.isdigit, raw_id))
            if not clean_id: clean_id = raw_id  # 如果没有数字，回退到原始ID

            # 目标文件夹: OUTPUT/fine-tuning/001/
            target_dir = os.path.join(OUTPUT_DIR, split_type, clean_id)
            os.makedirs(target_dir, exist_ok=True)

            print(f"   Sample: {sample_name} -> ID: {clean_id}")

            # --- A. 处理图像 (Image) ---
            img_in_path = os.path.join(images_in, sample_name)

            # 【合并逻辑】直接定义我们要的最终文件名 (.img.nii.gz)
            img_out_name = f"{clean_id}.img.nii.gz"
            img_out_path = os.path.join(target_dir, img_out_name)

            arr, meta = build_volume(img_in_path)
            if arr is None:
                print(f"      [Error] No slices found for {sample_name}, skipping.")
                continue

            sitk_img = array_to_sitk(arr, meta)
            # 使用集成的安全保存函数
            save_nifti_safe(sitk_img, img_out_path)

            # --- B. 处理标签 (Label) ---
            lbl_in_path = os.path.join(labels_in, sample_name)
            if os.path.exists(lbl_in_path):
                # 【合并逻辑】直接定义我们要的最终文件名 (.label.nii.gz)
                lbl_out_name = f"{clean_id}.label.nii.gz"
                lbl_out_path = os.path.join(target_dir, lbl_out_name)

                l_arr, l_meta = build_volume(lbl_in_path)
                if l_arr is not None:
                    # 标签必须转为 uint8 整数
                    sitk_lbl = array_to_sitk(l_arr.astype(np.uint8), l_meta)
                    save_nifti_safe(sitk_lbl, lbl_out_path)
            else:
                print(f"      [Warning] No label found for {sample_name}")

    print(f"\n[Success] All tasks finished.")
    print(f"Check output at: {OUTPUT_DIR}")
    print(f"Structure example: {OUTPUT_DIR}/fine-tuning/001/001.img.nii.gz")


if __name__ == "__main__":
    main()