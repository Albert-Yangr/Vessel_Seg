import os
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm

# ================= 1. 深度配置 =================
CONFIG = {
    # 原始数据根目录 (包含 PA000XXX 文件夹)
    "source_root": "/home/yangrui/Project/Base-model/datasets/Parse/Parse-origin/train",
    # 目标裁剪目录
    "target_root": "/home/yangrui/Project/Base-model/datasets/Parse/Parse-reshape/all",

}


# ================= 2. 修改后的核心函数 =================

def copy_without_cropping(img_src, lab_src, case_dst_dir, pid):
    """直接复制文件而不进行裁剪处理"""
    try:
        # 目标文件名: 5.img.nii.gz / 5.label.nii.gz
        final_img_name = f"{pid}.img.nii.gz"
        final_lab_name = f"{pid}.label.nii.gz"

        final_img_path = os.path.join(case_dst_dir, final_img_name)
        final_lab_path = os.path.join(case_dst_dir, final_lab_name)

        # 直接复制文件而不是裁剪和重新保存
        shutil.copy2(img_src, final_img_path)
        shutil.copy2(lab_src, final_lab_path)

        return True
    except Exception as e:
        print(f"❌ 文件复制失败: {e}")
        return False


# ================= 3. 执行主逻辑 =================

def main():
    src_root = CONFIG["source_root"]
    dst_root = CONFIG["target_root"]

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # 获取所有病例文件夹
    all_files = os.listdir(src_root)
    patient_folders = sorted([f for f in all_files if os.path.isdir(os.path.join(src_root, f))])

    print(f"🚀 开始重建文件结构，检测到 {len(patient_folders)} 个病例...")
    print("📝 模式: 仅重建文件结构，不进行数据裁剪")

    success_count = 0
    skip_count = 0

    for folder in tqdm(patient_folders):
        # 1. 解析 ID (PA000005 -> 5)
        try:
            pid_int = int(folder.replace("PA", ""))
            pid = str(pid_int)
        except:
            pid = folder

        # 2. 源文件路径
        img_src = os.path.join(src_root, folder, "image", f"{folder}.nii.gz")
        lab_src = os.path.join(src_root, folder, "label", f"{folder}.nii.gz")

        # 检查源文件是否存在
        if not os.path.exists(img_src) or not os.path.exists(lab_src):
            print(f"⚠️  病例 {pid} 源文件不存在，跳过")
            skip_count += 1
            continue

        # 3. 创建目标目录 (parse-clip/5/)
        case_dst_dir = os.path.join(dst_root, pid)
        os.makedirs(case_dst_dir, exist_ok=True)

        # 4. 直接复制文件而不进行裁剪
        if copy_without_cropping(img_src, lab_src, case_dst_dir, pid):
            success_count += 1
        else:
            skip_count += 1

    print(f"\n✅ 文件结构重建完成！")
    print(f"📊 统计信息:")
    print(f"   - 成功处理: {success_count} 个病例")
    print(f"   - 跳过: {skip_count} 个病例")
    print(f"   - 结果保存在: {dst_root}")
    print(f"   - 文件命名格式: {{PID}}.img.nii.gz 和 {{PID}}.label.nii.gz")


if __name__ == "__main__":
    main()