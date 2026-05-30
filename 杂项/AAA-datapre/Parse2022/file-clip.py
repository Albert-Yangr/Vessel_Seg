import os
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm

# ================= 1. 深度配置 =================
CONFIG = {
    # 原始数据根目录 (包含 PA000XXX 文件夹)
    "source_root": "/home/yangrui/Project/Base-model/datasets/Parse2022/train",
    # 目标裁剪目录
    "target_root": "/home/yangrui/Project/Base-model/datasets/Parse2022/parse-clip/all",
    # 裁剪留白 (像素)
    "margin": 0
}


# ================= 2. 核心算法函数 =================

def get_foreground_bbox(label_data, margin=5):
    """计算前景边界框坐标"""
    coords = np.argwhere(label_data > 0)
    if coords.size == 0:
        return None

    # 基础边界
    min_idx = np.maximum(coords.min(axis=0) - margin, 0)
    max_idx = np.minimum(coords.max(axis=0) + 1 + margin, label_data.shape)

    return [slice(min_idx[i], max_idx[i]) for i in range(3)], min_idx


def update_affine_offset(affine, min_indices):
    """更新仿射矩阵以保持空间坐标一致性"""
    new_affine = affine.copy()
    offset_vec = np.array([min_indices[0], min_indices[1], min_indices[2], 1])
    new_origin = affine @ offset_vec
    new_affine[:3, 3] = new_origin[:3]
    return new_affine


# ================= 3. 执行主逻辑 =================

def main():
    src_root = CONFIG["source_root"]
    dst_root = CONFIG["target_root"]

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # --- 修复 SyntaxError: 不再使用推导式内的赋值表达式 ---
    all_files = os.listdir(src_root)
    patient_folders = sorted([f for f in all_files if os.path.isdir(os.path.join(src_root, f))])

    print(f"🚀 开始处理，检测到 {len(patient_folders)} 个病例...")

    for folder in tqdm(patient_folders):
        # 1. 解析 ID (PA000005 -> 5)
        try:
            pid_int = int(folder.replace("PA", ""))
            pid = str(pid_int)
        except:
            pid = folder

        # 2. 路径对齐: /PA000005/image/PA000005.nii.gz
        img_src = os.path.join(src_root, folder, "image", f"{folder}.nii.gz")
        lab_src = os.path.join(src_root, folder, "label", f"{folder}.nii.gz")

        if not os.path.exists(img_src) or not os.path.exists(lab_src):
            continue

        # 3. 创建目标目录 (parse-clip/5/)
        case_dst_dir = os.path.join(dst_root, pid)
        os.makedirs(case_dst_dir, exist_ok=True)

        try:
            # 4. 读取与裁剪
            img_obj = nib.load(img_src)
            lab_obj = nib.load(lab_src)

            img_data = img_obj.get_fdata()
            lab_data = lab_obj.get_fdata()

            slices, min_idx = get_foreground_bbox(lab_data, CONFIG["margin"])
            if slices is None: continue

            cropped_img = img_data[tuple(slices)]
            cropped_lab = lab_data[tuple(slices)]

            # 5. 坐标校正
            new_affine = update_affine_offset(img_obj.affine, min_idx)

            # 6. 强制命名防御：先存为标准 nii.gz，再重命名
            # 目标文件名: 5.img.nii.gz / 5.label.nii.gz
            final_img_name = f"{pid}.img.nii.gz"
            final_lab_name = f"{pid}.label.nii.gz"

            temp_img_p = os.path.join(case_dst_dir, f"tmp_save_{pid}_img.nii.gz")
            temp_lab_p = os.path.join(case_dst_dir, f"tmp_save_{pid}_lab.nii.gz")

            # 保存为 float32 图像和 uint8 标签
            nib.save(nib.Nifti1Image(cropped_img.astype(np.float32), new_affine, img_obj.header), temp_img_p)
            nib.save(nib.Nifti1Image(cropped_lab.astype(np.uint8), new_affine, lab_obj.header), temp_lab_p)

            # 7. 强制系统级重命名，确保文件名完全一致
            os.rename(temp_img_p, os.path.join(case_dst_dir, final_img_name))
            os.rename(temp_lab_p, os.path.join(case_dst_dir, final_lab_name))

        except Exception as e:
            print(f"❌ 病例 {pid} 处理失败: {e}")

    print(f"\n✅ 任务完成！结果保存在: {dst_root}")


if __name__ == "__main__":
    main()