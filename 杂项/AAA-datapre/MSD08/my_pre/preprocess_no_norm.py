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
OUTPUT_DIR = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-61-noclip/all"

# 3. 筛选标准 (只处理层厚 < 2.0mm 的数据)
THIN_SLICE_THRESHOLD = 2.0

# 4. 标签前缀 (用于匹配文件名)
SRC_LABEL_PREFIX = "hp"

# 5. 固定截断设置 (Fixed Clipping)
ENABLE_CLIPPING = True
CLIP_MIN = -100.0
CLIP_MAX = 250.0

# 6. Z轴两倍缩放设置
ENABLE_Z_RESCALE = False

# 7. 【新增开关】是否开启前景 ROI 裁剪
#    True:  根据标签自动裁剪掉无关背景（只保留血管/肝脏区域）
#    False: 保留原始 512x512 的完整视野进行处理
ENABLE_ROI_CROP = False


# ================= 核心函数 =================

def extract_id(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None


def save_nifti_safe(image_obj, final_path):
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


def resample_z_axis_x2(itk_image, is_label=False):
    orig_spacing = itk_image.GetSpacing()
    orig_size = itk_image.GetSize()
    new_spacing = (orig_spacing[0], orig_spacing[1], orig_spacing[2] * 0.5)
    new_size = [
        int(orig_size[0]),
        int(orig_size[1]),
        int(round(orig_size[2] * (orig_spacing[2] / new_spacing[2])))
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(itk_image)


def process_single_case(img_path, lbl_path, output_folder, case_id):
    image = sitk.ReadImage(img_path)
    label = sitk.ReadImage(lbl_path)

    spacing = image.GetSpacing()
    z_spacing = spacing[2]
    if z_spacing > THIN_SLICE_THRESHOLD:
        return False, f"Skip (Thick slice: {z_spacing:.2f}mm)"

    # --- 裁剪逻辑控制 ---
    if ENABLE_ROI_CROP:
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        binary_label = sitk.BinaryThreshold(label, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
        label_stats.Execute(binary_label)
        if not label_stats.HasLabel(1):
            return False, "Skip (Empty Label)"
        bbox = label_stats.GetBoundingBox(1)
        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetRegionOfInterest(bbox)
        working_img_obj = roi_filter.Execute(image)
        working_lbl_obj = roi_filter.Execute(label)
        crop_msg = "ROI Cropped"
    else:
        working_img_obj = image
        working_lbl_obj = label
        crop_msg = "Full View (No Crop)"

    # --- 重采样 ---
    if ENABLE_Z_RESCALE:
        processed_image_obj = resample_z_axis_x2(working_img_obj, is_label=False)
        processed_label_obj = resample_z_axis_x2(working_lbl_obj, is_label=True)
        rescale_msg = f"Z-Rescaled"
    else:
        processed_image_obj = working_img_obj
        processed_label_obj = working_lbl_obj
        rescale_msg = "No Rescale"

    # --- 像素处理 ---
    img_arr = sitk.GetArrayFromImage(processed_image_obj)
    lbl_arr = sitk.GetArrayFromImage(processed_label_obj)

    new_lbl_arr = np.zeros_like(lbl_arr)
    new_lbl_arr[lbl_arr > 0] = 1

    img_arr = img_arr.astype(np.float32)
    if ENABLE_CLIPPING:
        img_arr = np.clip(img_arr, CLIP_MIN, CLIP_MAX)
        clip_msg = f"Clipped"
    else:
        clip_msg = "No Clip"

    # --- 写回并保存 ---
    final_img_obj = sitk.GetImageFromArray(img_arr)
    final_img_obj.CopyInformation(processed_image_obj)
    final_lbl_obj = sitk.GetImageFromArray(new_lbl_arr.astype(np.uint8))
    final_lbl_obj.CopyInformation(processed_label_obj)

    case_dir = os.path.join(output_folder, case_id)
    os.makedirs(case_dir, exist_ok=True)
    save_nifti_safe(final_img_obj, os.path.join(case_dir, f"{case_id}.img.nii.gz"))
    save_nifti_safe(final_lbl_obj, os.path.join(case_dir, f"{case_id}.label.nii.gz"))

    return True, f"Success ({crop_msg}, {clip_msg}, {rescale_msg}, Shape: {img_arr.shape})"


# ================= 主程序 (省略部分打印逻辑以保持简洁) =================
def main():
    img_files = sorted(glob.glob(os.path.join(SRC_IMG_DIR, "hepaticvessel_*.nii.gz")))
    print(f"⚙️  ROI 裁剪开关: {'开启' if ENABLE_ROI_CROP else '关闭'}")

    count_processed = 0
    for img_path in img_files:
        case_id = extract_id(os.path.basename(img_path))
        lbl_path = os.path.join(SRC_LABEL_DIR, f"{SRC_LABEL_PREFIX}{case_id}.nii.gz")
        if not os.path.exists(lbl_path): continue

        success, msg = process_single_case(img_path, lbl_path, OUTPUT_DIR, case_id)
        if success:
            print(f"✅ [ID: {case_id}] {msg}")
            count_processed += 1


if __name__ == "__main__":
    main()