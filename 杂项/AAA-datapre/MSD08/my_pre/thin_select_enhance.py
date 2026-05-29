import os
import shutil
import glob
import re
import numpy as np
import SimpleITK as sitk

# ================= 配置区域 =================

# 1. 输入路径 (原始数据)
SRC_IMG_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/imagesTr"
SRC_LABEL_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/reannotated_fixed"

# 2. 输出路径
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-61-nopre/all"

# 3. 筛选标准 (只处理层厚 < 2.0mm 的数据)
THIN_SLICE_THRESHOLD = 2.0

# 4. 标签前缀 (用于匹配文件名)
SRC_LABEL_PREFIX = "hp"

# 5. 【新增】是否开启前景裁剪开关
# True: 执行基于真值的 ROI 裁剪 (训练数据常用)
# False: 保留原始图像尺寸，不裁剪 (推理或保留背景时用)
ENABLE_FOREGROUND_CROP = True  # <--- 修改点：设置为 False 则关闭裁剪


# ================= 核心函数 =================

def extract_id(filename):
    """从文件名中提取数字 ID"""
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None


def save_nifti_safe(image_obj, final_path):
    """安全保存函数"""
    final_path = str(final_path)
    dirname = os.path.dirname(final_path)
    filename = os.path.basename(final_path)

    temp_filename = "TEMP_" + filename.replace(".", "_") + ".nii.gz"
    temp_path = os.path.join(dirname, temp_filename)

    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(temp_path)
        writer.SetImageIO("NiftiImageIO")
        writer.Execute(image_obj)

        if os.path.exists(final_path):
            os.remove(final_path)
        shutil.move(temp_path, final_path)

        junk_base = final_path.replace(".nii.gz", "")
        for ext in [".hdr", ".img"]:
            junk_file = junk_base + ext
            if os.path.exists(junk_file):
                os.remove(junk_file)

    except Exception as e:
        print(f"      ❌ 保存失败: {filename} -> {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_intensity(img_arr):
    """执行 20-98 分位数截断，并归一化到 [0, 1]"""
    lower = np.percentile(img_arr, 20)
    upper = np.percentile(img_arr, 98)
    img_arr = np.clip(img_arr, lower, upper)
    if upper != lower:
        img_arr = (img_arr - lower) / (upper - lower)
    else:
        img_arr[:] = 0
    return img_arr


def process_single_case(img_path, lbl_path, output_folder, case_id):
    """处理单个样本"""

    # 1. 读取图像和标签
    image = sitk.ReadImage(img_path)
    label = sitk.ReadImage(lbl_path)

    # 2. 【核心筛选】检查层厚
    spacing = image.GetSpacing()
    z_spacing = spacing[2]

    if z_spacing > THIN_SLICE_THRESHOLD:
        return False, f"Skip (Thick slice: {z_spacing:.2f}mm)"

    # 3. 【裁剪逻辑修改】根据开关决定是否裁剪
    # 定义两个变量用于存放待处理的图像对象
    img_obj_to_process = None
    lbl_obj_to_process = None

    if ENABLE_FOREGROUND_CROP:  # <--- 修改点：如果不开启，直接跳过此块
        # --- 开启裁剪模式 ---
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        # 二值化用于计算 bbox
        binary_temp = sitk.BinaryThreshold(label, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
        label_stats.Execute(binary_temp)

        if not label_stats.HasLabel(1):
            return False, "Skip (Empty Label)"

        bbox = label_stats.GetBoundingBox(1)

        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetRegionOfInterest(bbox)

        img_obj_to_process = roi_filter.Execute(image)
        lbl_obj_to_process = roi_filter.Execute(label)
    else:
        # --- 关闭裁剪模式 --- <--- 修改点
        # 直接使用原始图像和标签
        img_obj_to_process = image
        lbl_obj_to_process = label

    # 4. 转为 Numpy 进行像素处理
    # 注意这里使用的是上一步决定好的对象 (裁剪后 或 原始)
    img_arr = sitk.GetArrayFromImage(img_obj_to_process)
    lbl_arr = sitk.GetArrayFromImage(lbl_obj_to_process)

    # 5. 【标签处理】二值化
    new_lbl_arr = np.zeros_like(lbl_arr)
    new_lbl_arr[lbl_arr > 0] = 1

    # 6. 【图像处理】归一化
    new_img_arr = normalize_intensity(img_arr)

    # 7. 转回 SimpleITK 对象
    final_img_obj = sitk.GetImageFromArray(new_img_arr)
    final_img_obj.CopyInformation(img_obj_to_process)  # 继承空间信息

    final_lbl_obj = sitk.GetImageFromArray(new_lbl_arr.astype(np.uint8))
    final_lbl_obj.CopyInformation(lbl_obj_to_process)  # 继承空间信息

    # 8. 保存
    case_dir = os.path.join(output_folder, case_id)
    os.makedirs(case_dir, exist_ok=True)

    target_img_name = f"{case_id}.img.nii.gz"
    target_lbl_name = f"{case_id}.label.nii.gz"

    save_nifti_safe(final_img_obj, os.path.join(case_dir, target_img_name))
    save_nifti_safe(final_lbl_obj, os.path.join(case_dir, target_lbl_name))

    # 返回信息中增加当前模式提示
    status_msg = "Cropped" if ENABLE_FOREGROUND_CROP else "Full-Size"
    return True, f"Success ({status_msg}, Shape: {new_img_arr.shape})"


# ================= 主程序 =================

def main():
    if not os.path.exists(SRC_IMG_DIR):
        print(f"❌ 源目录不存在: {SRC_IMG_DIR}")
        return

    img_files = sorted(glob.glob(os.path.join(SRC_IMG_DIR, "hepaticvessel_*.nii.gz")))

    print(f"🔍 扫描目录: {SRC_IMG_DIR}")
    print(f"📄 找到文件: {len(img_files)} 个")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"⚙️ 筛选条件: 层厚 < {THIN_SLICE_THRESHOLD} mm")
    # 打印当前的裁剪配置状态
    print(f"✂️ 前景裁剪: {'✅ 开启 (基于真值)' if ENABLE_FOREGROUND_CROP else '⛔ 关闭 (保留原图)'}")  # <--- 修改点
    print("-" * 60)

    count_processed = 0
    count_skipped_thick = 0

    for img_path in img_files:
        filename = os.path.basename(img_path)
        case_id = extract_id(filename)

        if not case_id:
            continue

        lbl_name = f"{SRC_LABEL_PREFIX}{case_id}.nii.gz"
        lbl_path = os.path.join(SRC_LABEL_DIR, lbl_name)

        if not os.path.exists(lbl_path):
            print(f"⚠️  [ID: {case_id}] 缺失标签文件，跳过")
            continue

        print(f"⏳ [ID: {case_id}] 处理中...", end="\r")
        success, msg = process_single_case(img_path, lbl_path, OUTPUT_DIR, case_id)

        if success:
            print(f"✅ [ID: {case_id}] {msg}")
            count_processed += 1
        else:
            if "Thick slice" in msg:
                count_skipped_thick += 1
            else:
                print(f"❌ [ID: {case_id}] {msg}")

    print("-" * 60)
    print(f"🎉 全部完成！")
    print(f"📥 总输入文件: {len(img_files)}")
    print(f"⏭️ 跳过厚层数据: {count_skipped_thick}")
    print(f"💾 成功处理并保存: {count_processed}")
    print(f"📂 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()