import os
import shutil
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk  # 引入 SimpleITK 用于读取尺寸


def reformat_dataset():
    # ================= 配置路径 =================
    # 原始数据根目录
    origin_root = Path("/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-origin")
    img_dir = origin_root / "data"
    mask_dir = origin_root / "mask"

    # 目标输出目录
    target_root = Path("/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-S/all")

    # ================= 开关设置 =================
    # True: 只转移 X轴 > 300 的数据 (过滤掉小图)
    # False: 转移所有数据 (不进行尺寸检查)
    ONLY_TRANSFER_LARGE_IMAGES = True
    # ===========================================

    # 1. 检查源目录是否存在
    if not img_dir.exists() or not mask_dir.exists():
        print(f"错误: 源目录不存在。\n请检查: {img_dir}\n或: {mask_dir}")
        return

    # 2. 创建目标根目录
    target_root.mkdir(parents=True, exist_ok=True)

    print(f"目标目录已创建/确认: {target_root}")
    if ONLY_TRANSFER_LARGE_IMAGES:
        print("⚡ 筛选模式已开启: 仅转移 X轴 > 300 的图像")
    else:
        print("📦 全量模式: 转移所有图像")

    # 3. 获取所有图像文件
    image_files = sorted(list(img_dir.glob("*.nii.gz")))
    print(f"找到 {len(image_files)} 个图像文件，开始处理...")

    success_count = 0
    error_count = 0
    skip_small_count = 0

    # 使用 tqdm 显示进度条
    for img_path in tqdm(image_files, desc="处理进度"):
        file_name = img_path.name  # 例如: "000.nii.gz"
        mask_path = mask_dir / file_name

        # 检查 Mask 是否存在
        if not mask_path.exists():
            # print(f"\n[跳过] 未找到对应的 Mask 文件: {file_name}")
            error_count += 1
            continue

        # --- 【新增】尺寸过滤逻辑 ---
        if ONLY_TRANSFER_LARGE_IMAGES:
            try:
                # 使用 ImageFileReader 只读取头信息，速度非常快
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(img_path))
                reader.ReadImageInformation()
                size = reader.GetSize()  # (X, Y, Z)

                # 如果 X轴 <= 300，跳过
                if size[0] <= 300:
                    # tqdm.write(f"🚫 [过滤] 跳过小图 X={size[0]}: {file_name}")
                    skip_small_count += 1
                    continue

            except Exception as e:
                print(f"\n[错误] 读取图像信息失败 {file_name}: {e}")
                error_count += 1
                continue
        # ---------------------------

        # --- ID 转换逻辑 ---
        raw_id = file_name.replace(".nii.gz", "")
        try:
            new_id = str(int(raw_id))
        except ValueError:
            new_id = raw_id

        # --- 创建目标文件夹 ---
        case_dir = target_root / new_id
        case_dir.mkdir(exist_ok=True)

        # --- 定义新文件名 ---
        new_img_name = f"{new_id}.img.nii.gz"
        new_mask_name = f"{new_id}.label.nii.gz"

        target_img_path = case_dir / new_img_name
        target_mask_path = case_dir / new_mask_name

        # --- 复制文件 ---
        try:
            shutil.copy2(img_path, target_img_path)
            shutil.copy2(mask_path, target_mask_path)
            success_count += 1
        except Exception as e:
            print(f"\n[错误] 复制文件失败 {file_name}: {e}")
            error_count += 1

    print("-" * 30)
    print(f"处理完成!")
    print(f"✅ 成功转移: {success_count}")
    if ONLY_TRANSFER_LARGE_IMAGES:
        print(f"🚫 过滤小图: {skip_small_count}")
    print(f"❌ 失败/缺失Mask: {error_count}")
    print(f"数据已保存至: {target_root}")


if __name__ == "__main__":
    reformat_dataset()