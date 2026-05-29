import os
import shutil
import time
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cupy as cp  # GPU 加速库
import cupyx.scipy.ndimage as cp_ndimage
from concurrent.futures import ProcessPoolExecutor

# ================= 1. 配置区域 =================
# 输入输出路径
SRC_ROOT = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-clip"
DST_ROOT = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-ROI"
SUBSETS = ["all"]

# 心脏 ROI 提取参数 (保持与您的 ROI_batch.py 一致)
PARAM_THRESH = -600
PARAM_ERODE = 12
PARAM_DILATE = 15
COPY_LABELS = True

# GPU 并行设置 (3D 图像非常占显存，建议 1-2，根据显存大小调整)
MAX_WORKERS = 2


# ===============================================

def create_gpu_ball(radius):
    """在 GPU 上生成球形结构元素"""
    z, y, x = cp.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 + z ** 2) <= radius ** 2


def extract_heart_advanced_gpu(input_path, output_path, label_path=None,
                               threshold_hu=-300,
                               erosion_radius=12,
                               dilation_radius=20):
    """
    心脏 ROI 提取 GPU 加速版：核心剥离 + 距离变换安全区
    """
    try:
        # 1. 读取数据并推送至 GPU
        img = nib.load(input_path)
        data_gpu = cp.asarray(img.get_fdata().astype(np.float32))
        affine = img.affine
        header = img.header

        # 2. 基础阈值掩码
        binary_mask = data_gpu > threshold_hu

        # 3. 核心分离 (GPU 腐蚀)
        selem_erode = create_gpu_ball(erosion_radius)
        eroded_mask = cp_ndimage.binary_erosion(binary_mask, structure=selem_erode)

        # 4. 提取最大连通域 (锁定心脏核心)
        labeled_mask, num_features = cp_ndimage.label(eroded_mask)
        if num_features == 0:
            return "Fail_Erosion"

        # 使用 GPU 计算各区域大小并获取最大标签
        counts = cp.bincount(labeled_mask.ravel())
        largest_label = cp.argmax(counts[1:]) + 1
        heart_core = (labeled_mask == largest_label)

        # 5. 建立安全区 (GPU 距离变换)
        # 对核心掩码取反，计算到核心的距离
        dist_map = cp_ndimage.distance_transform_edt(cp.logical_not(heart_core))
        safety_zone = dist_map <= dilation_radius

        # 6. 最终切割与闭运算
        final_mask = cp.logical_and(binary_mask, safety_zone)


        # === 7. 标签无损检验模块 (GPU 加速) ===
        check_msg = ""
        if label_path and os.path.exists(label_path):
            lab_data_gpu = cp.asarray(nib.load(label_path).get_fdata() > 0)
            vessel_count = cp.sum(lab_data_gpu)

            if vessel_count > 0:
                preserved_vessel = cp.logical_and(lab_data_gpu, final_mask)
                preserved_count = cp.sum(preserved_vessel)
                loss_count = int(vessel_count - preserved_count)
                preservation_rate = (preserved_count / vessel_count) * 100

                if loss_count > 0:
                    check_msg = f" | [警告] 标签受损! 丢失:{loss_count}体素({preservation_rate:.2f}%)"
                else:
                    check_msg = " | [安全] 100%保留"
        # =================================================================

        # 8. 应用遮罩 (-1024) 并传回 CPU
        masked_data = data_gpu.copy()
        masked_data[cp.logical_not(final_mask)] = -1000
        final_data_cpu = masked_data.get()


        # 9. 保存结果
        new_img = nib.Nifti1Image(final_data_cpu, affine, header)
        nib.save(new_img, output_path)

        # 显式清理显存
        cp.get_default_memory_pool().free_all_blocks()

        return f"Success{check_msg}"

    except Exception as e:
        return f"Error: {str(e)}"


def process_single_case(args):
    """单病例调度逻辑"""
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

    if os.path.exists(src_img_path):
        if not os.path.exists(dst_img_path):
            status = extract_heart_advanced_gpu(src_img_path, dst_img_path,
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
    print("=== 开始批量心脏ROI提取 (GPU加速版) ===")
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
    # 根据显存大小调整并行数
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_single_case, tasks)

    print("\n" + "=" * 30)
    print(f"处理完成！总耗时: {time.time() - start_global:.2f} 秒")


if __name__ == "__main__":
    main()