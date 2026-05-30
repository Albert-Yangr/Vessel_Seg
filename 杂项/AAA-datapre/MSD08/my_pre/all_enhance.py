import os
import shutil
import glob
import re
import numpy as np
import SimpleITK as sitk

# ================= 配置区域 =================

# 1. 输入路径
SRC_IMG_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/imagesTr"
SRC_LABEL_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8/reannotated_fixed"

# 2. 输出路径
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-ada/test"

# 3. 判定标准与参数
THICKNESS_THRESHOLD = 2.1  # 大于等于此值则认为是厚层
Z_UPSAMPLE_FACTOR = 4  # 厚层数据的Z轴放大倍数

# 4. 标签前缀匹配 (例如 image: hepaticvessel_001 -> label: hp001)
SRC_LABEL_PREFIX = "hp"


# ================= 工具函数 =================

def extract_id(filename):
    """从文件名提取数字ID"""
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None


def save_nifti_safe(image_obj, final_path):
    """安全保存 NIfTI，防止生成 .hdr/.img"""
    final_path = str(final_path)
    dirname = os.path.dirname(final_path)
    filename = os.path.basename(final_path)

    # 使用临时文件名
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

        # 清理可能产生的垃圾文件
        junk_base = final_path.replace(".nii.gz", "")
        for ext in [".hdr", ".img"]:
            junk_file = junk_base + ext
            if os.path.exists(junk_file):
                os.remove(junk_file)

    except Exception as e:
        print(f"      ❌ Save Error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_intensity(img_arr):
    """20-98 分位数截断并归一化到 0-1"""
    lower = np.percentile(img_arr, 20)
    upper = np.percentile(img_arr, 98)
    img_arr = np.clip(img_arr, lower, upper)
    if upper != lower:
        img_arr = (img_arr - lower) / (upper - lower)
    else:
        img_arr[:] = 0
    return img_arr


def upsample_sitk(itk_img, factor, is_label=False):
    """SimpleITK Z轴上采样"""
    orig_size = itk_img.GetSize()
    orig_spacing = itk_img.GetSpacing()

    # Z轴尺寸变大，Spacing变小
    new_size = [orig_size[0], orig_size[1], int(orig_size[2] * factor)]
    new_spacing = [orig_spacing[0], orig_spacing[1], orig_spacing[2] / factor]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampler.SetOutputDirection(itk_img.GetDirection())

    # 标签用最近邻，图像用线性
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(itk_img)


# ================= 核心逻辑 =================

def process_single_case(img_path, lbl_path, output_folder, case_id):
    # 1. 读取
    image = sitk.ReadImage(img_path)
    label = sitk.ReadImage(lbl_path)

    # 2. 检查层厚
    spacing = image.GetSpacing()
    z_spacing = spacing[2]

    # 3. ROI 裁剪 (Crop)
    # 先裁剪可以显著减少后续计算量，且去除了背景干扰
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    binary_label_for_bbox = sitk.BinaryThreshold(label, lowerThreshold=1, upperThreshold=255, insideValue=1,
                                                 outsideValue=0)
    label_stats.Execute(binary_label_for_bbox)

    if not label_stats.HasLabel(1):
        return False, "Skip (Empty Label)"

    bbox = label_stats.GetBoundingBox(1)
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetRegionOfInterest(bbox)

    cropped_image = roi_filter.Execute(image)
    cropped_label = roi_filter.Execute(label)

    process_info = ""

    # 4. 条件分支：是否需要上采样
    if z_spacing >= THICKNESS_THRESHOLD:
        # === 厚层：执行上采样 ===
        final_image_obj = upsample_sitk(cropped_image, factor=Z_UPSAMPLE_FACTOR, is_label=False)
        final_label_obj = upsample_sitk(cropped_label, factor=Z_UPSAMPLE_FACTOR, is_label=True)
        process_info = f"Upsampled x{Z_UPSAMPLE_FACTOR} (Thick: {z_spacing:.2f}mm)"
    else:
        # === 薄层：保持原样 ===
        final_image_obj = cropped_image
        final_label_obj = cropped_label
        process_info = f"Kept Original (Thin: {z_spacing:.2f}mm)"

    # 5. Numpy 处理 (归一化 + 标签二值化)
    img_arr = sitk.GetArrayFromImage(final_image_obj)
    lbl_arr = sitk.GetArrayFromImage(final_label_obj)

    # 标签二值化
    new_lbl_arr = np.zeros_like(lbl_arr)
    new_lbl_arr[lbl_arr > 0] = 1

    # 图像归一化
    new_img_arr = normalize_intensity(img_arr)

    # 6. 重建 SimpleITK 对象 (保留空间信息)
    out_img_obj = sitk.GetImageFromArray(new_img_arr)
    out_img_obj.CopyInformation(final_image_obj)

    out_lbl_obj = sitk.GetImageFromArray(new_lbl_arr.astype(np.uint8))
    out_lbl_obj.CopyInformation(final_label_obj)

    # 7. 保存
    case_dir = os.path.join(output_folder, case_id)
    os.makedirs(case_dir, exist_ok=True)

    save_nifti_safe(out_img_obj, os.path.join(case_dir, f"{case_id}.img.nii.gz"))
    save_nifti_safe(out_lbl_obj, os.path.join(case_dir, f"{case_id}.label.nii.gz"))

    return True, process_info


# ================= 主程序 =================

def main():
    if not os.path.exists(SRC_IMG_DIR):
        print("❌ 源目录不存在")
        return

    img_files = sorted(glob.glob(os.path.join(SRC_IMG_DIR, "hepaticvessel_*.nii.gz")))

    print(f"🔍 发现文件: {len(img_files)} 个")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"⚙️  策略: 层厚 >= {THICKNESS_THRESHOLD}mm -> 放大 {Z_UPSAMPLE_FACTOR}倍 | 其他 -> 仅裁剪归一化")
    print("-" * 60)

    count_upsampled = 0
    count_kept = 0

    for img_path in img_files:
        filename = os.path.basename(img_path)
        case_id = extract_id(filename)

        if not case_id: continue

        lbl_name = f"{SRC_LABEL_PREFIX}{case_id}.nii.gz"
        lbl_path = os.path.join(SRC_LABEL_DIR, lbl_name)

        if not os.path.exists(lbl_path):
            print(f"⚠️  [ID: {case_id}] 缺标签，跳过")
            continue

        print(f"⏳ [ID: {case_id}] 处理中...", end="\r")

        success, msg = process_single_case(img_path, lbl_path, OUTPUT_DIR, case_id)

        if success:
            print(f"✅ [ID: {case_id}] {msg}")
            if "Upsampled" in msg:
                count_upsampled += 1
            else:
                count_kept += 1
        else:
            print(f"❌ [ID: {case_id}] {msg}")

    print("-" * 60)
    print(f"🎉 处理完成")
    print(f"📈 膨胀(Upsample) 数量: {count_upsampled}")
    print(f"⏹️ 保持(Original) 数量: {count_kept}")
    print(f"📂 结果路径: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()