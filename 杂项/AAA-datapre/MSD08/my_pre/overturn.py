import os
import shutil
from pathlib import Path
import SimpleITK as sitk
import numpy as np

# ==========================细化的标签x轴颠倒了，需要重新翻转=======================================


# ================= 配置区域 =================
# 原始标签文件夹 (你的新标签)
SRC_LABEL_DIR = Path("/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/reannotated")

# 输出文件夹 (自动创建，处理好的文件放这里)
DST_LABEL_DIR = Path("/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/reannotated_fixed")

# 翻转轴设置
# 也就是 Numpy 数组的 (z, y, x) 对应的索引
# 通常医学图像中：
# axis=2 是 x轴 (左右) -> 我们要翻转这个
# axis=1 是 y轴 (前后)
# axis=0 是 z轴 (上下)
FLIP_AXIS = 2


# ===========================================

def flip_and_save(file_path, save_path):
    try:
        # 1. 读取原始标签
        label_img = sitk.ReadImage(str(file_path))

        # 2. 转为 Numpy 数组 [z, y, x]
        arr = sitk.GetArrayFromImage(label_img)

        # 3. 执行翻转 (Flip)
        # np.flip(arr, axis=2) 表示在 X 轴方向做镜像
        arr_flipped = np.flip(arr, axis=FLIP_AXIS)

        # 4. 转回 SimpleITK 对象
        new_label = sitk.GetImageFromArray(arr_flipped)

        # 5. 🚨关键步骤：复制原始的空间信息🚨
        # 这保证了翻转后的像素矩阵，依然呆在原来的物理坐标框里
        new_label.CopyInformation(label_img)

        # 6. 保存
        sitk.WriteImage(new_label, str(save_path))
        print(f"✅ 已翻转并保存: {file_path.name}")
        return True

    except Exception as e:
        print(f"❌ 处理失败 {file_path.name}: {e}")
        return False


def main():
    if not SRC_LABEL_DIR.exists():
        print(f"❌ 错误：源目录不存在 {SRC_LABEL_DIR}")
        return

    # 创建输出目录
    DST_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 源目录: {SRC_LABEL_DIR}")
    print(f"📂 输出目录: {DST_LABEL_DIR}")
    print("-" * 50)

    # 查找所有 .nii.gz 文件
    files = sorted(list(SRC_LABEL_DIR.glob("*.nii.gz")))

    if not files:
        print("⚠️ 目录为空，未找到 .nii.gz 文件")
        return

    success_count = 0

    for file_path in files:
        save_path = DST_LABEL_DIR / file_path.name
        if flip_and_save(file_path, save_path):
            success_count += 1

    print("-" * 50)
    print(f"🎉 处理完成！共修复 {success_count} 个文件。")
    print(f"💡 请使用 ITK-SNAP 打开新旧文件对比，确认方向正确后，")
    print(f"   再修改后续的预处理脚本读取 '{DST_LABEL_DIR.name}' 文件夹。")


if __name__ == "__main__":
    main()