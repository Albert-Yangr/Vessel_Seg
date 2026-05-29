import os
import shutil
import time
import numpy as np
import nibabel as nib
from skimage import morphology, measure
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

# ================= 配置区域 =================
SRC_ROOT = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-clip"
DST_ROOT = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-ROI"
SUBSETS = ["test"]

PARAM_THRESH = -300
PARAM_ERODE = 12
PARAM_DILATE = 20
COPY_LABELS = True


# ===========================================

def extract_heart_advanced(input_path, output_path, label_path=None,
                           threshold_hu=-300,
                           erosion_radius=12,
                           dilation_radius=20):
    """
    带标签无损检验的核心处理算法
    """
    try:
        # 1. 读取数据
        img = nib.load(input_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # 2. 基础阈值
        binary_mask = data > threshold_hu

        # 3. 核心分离 (腐蚀)
        selem_erode = morphology.ball(erosion_radius)
        eroded_mask = morphology.binary_erosion(binary_mask, selem_erode)

        # 4. 提取最大连通域
        labeled_mask, num_features = ndimage.label(eroded_mask)
        if num_features == 0:
            return "Fail_Erosion"

        region_sizes = [(i, np.sum(labeled_mask == i)) for i in range(1, num_features + 1)]
        region_sizes.sort(key=lambda x: x[1], reverse=True)
        largest_label = region_sizes[0][0]
        heart_core = (labeled_mask == largest_label)

        # 5. 建立安全区 (距离变换)
        dist_map = ndimage.distance_transform_edt(~heart_core)
        safety_zone = dist_map <= dilation_radius

        # 6. 最终切割
        final_mask = np.logical_and(binary_mask, safety_zone)
        final_mask = morphology.binary_closing(final_mask, morphology.ball(2))

        # === 7. 【新增：标签无损检验模块】 ===
        check_msg = ""
        if label_path and os.path.exists(label_path):
            lab_data = nib.load(label_path).get_fdata()
            vessel_voxels = (lab_data > 0)
            vessel_count = np.sum(vessel_voxels)

            if vessel_count > 0:
                # 计算在 final_mask 内的标签体素
                preserved_vessel = np.logical_and(vessel_voxels, final_mask)
                preserved_count = np.sum(preserved_vessel)

                loss_count = vessel_count - preserved_count
                preservation_rate = (preserved_count / vessel_count) * 100

                if loss_count > 0:
                    check_msg = f" | [警告] 标签受损! 丢失:{loss_count}体素({preservation_rate:.2f}%)"
                else:
                    check_msg = " | [安全] 标签100%保留"
        # =====================================

        # 8. 应用遮罩 (-1024)
        masked_data = data.copy()
        masked_data[~final_mask] = -1024

        # 9. 保存
        new_img = nib.Nifti1Image(masked_data, affine, header)
        nib.save(new_img, output_path)
        return f"Success{check_msg}"

    except Exception as e:
        return f"Error: {str(e)}"


def process_single_case(args):
    case_path, dst_subset_dir, subset_name = args
    case_name = os.path.basename(case_path)

    img_name = f"{case_name}.img.nii.gz"
    label_name = f"{case_name}.label.nii.gz"

    src_img_path = os.path.join(case_path, img_name)
    src_label_path = os.path.join(case_path, label_name)

    dst_case_dir = os.path.join(dst_subset_dir, case_name)
    os.makedirs(dst_case_dir, exist_ok=True)

    dst_img_path = os.path.join(dst_case_dir, img_name)
    dst_label_path = os.path.join(dst_case_dir, label_name)

    # 处理图像并传入标签路径进行检验
    if os.path.exists(src_img_path):
        if not os.path.exists(dst_img_path):
            status = extract_heart_advanced(src_img_path, dst_img_path,
                                            label_path=src_label_path,
                                            threshold_hu=PARAM_THRESH,
                                            erosion_radius=PARAM_ERODE,
                                            dilation_radius=PARAM_DILATE)
            print(f"进度: {subset_name}/{case_name} -> {status}")
        else:
            print(f"跳过: {subset_name}/{case_name} (已存在)")

    if COPY_LABELS and os.path.exists(src_label_path):
        if not os.path.exists(dst_label_path):
            shutil.copy(src_label_path, dst_label_path)


def main():
    print("=== 开始批量心脏ROI提取 (含标签无损验证) ===")
    tasks = []
    for subset in SUBSETS:
        src_subset_path = os.path.join(SRC_ROOT, subset)
        dst_subset_path = os.path.join(DST_ROOT, subset)
        if not os.path.exists(src_subset_path): continue

        case_folders = [f for f in os.listdir(src_subset_path) if os.path.isdir(os.path.join(src_subset_path, f))]
        try:
            case_folders.sort(key=lambda x: int(x))
        except:
            case_folders.sort()

        for case_name in case_folders:
            case_path = os.path.join(src_subset_path, case_name)
            tasks.append((case_path, dst_subset_path, subset))

    start_global = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_single_case, tasks)

    print("\n" + "=" * 30)
    print(f"处理完成！总耗时: {time.time() - start_global:.2f} 秒")


if __name__ == "__main__":
    main()