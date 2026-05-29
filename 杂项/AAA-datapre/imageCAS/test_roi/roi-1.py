import os
import glob
import math
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm
import multiprocessing as mp
import time

# =========================================================================
#                               【参数控制台】
# =========================================================================

# 输入/输出 根目录
INPUT_ROOT = r"/home/yangrui/Project/Base-model/datasets/imageCAS/ImageCAS-original/all/801-1000"
OUTPUT_ROOT = r"/home/yangrui/Project/Base-model/datasets/imageCAS/new-ROI/all"

# 背景填充值
BG_VALUE = -1000

# --- 【修改处】范围控制：只处理 900 到 1000 ---
START_ID = 901
END_ID = 1000

# 算法参数
LUNG_THRESH = -800
ISOLATION_EROSION = 60
RESTORE_DILATION = 70

# 裁剪参数
CROP_PADDING = 0

# 多卡配置
VISIBLE_DEVICES = [0, 1, 3, 4, 5, 6, 7, 2]


# =========================================================================
#                             核心算法 (保持不变)
# =========================================================================

def get_lung_mask_cpu(img_arr):
    """CPU端提取肺部"""
    binary = img_arr < LUNG_THRESH
    labels, num = ndimage.label(binary)
    z, y, x = img_arr.shape

    corners = [(0, 0, 0), (0, 0, x - 1), (0, y - 1, 0), (z - 1, 0, 0)]
    bg_labels = set()
    for c in corners:
        if labels[c] != 0: bg_labels.add(labels[c])
    mask_no_bg = np.isin(labels, list(bg_labels), invert=True) & (labels != 0)

    lung_labels, lung_num = ndimage.label(mask_no_bg)
    if lung_num == 0: return np.zeros_like(img_arr, dtype=bool)
    sizes = ndimage.sum(mask_no_bg, lung_labels, range(1, lung_num + 1))
    valid_labels = np.where(sizes > (img_arr.size * 0.01))[0] + 1
    lung_mask = np.isin(lung_labels, valid_labels)

    for i in range(z):
        lung_mask[i] = ndimage.binary_fill_holes(lung_mask[i])
    struct = ndimage.generate_binary_structure(3, 1)
    lung_mask = ndimage.binary_dilation(lung_mask, structure=struct, iterations=3)
    return lung_mask


def process_shrink_wrap_gpu(img_arr, lung_mask, gpu_lib):
    """GPU加速负压贴合"""
    cp = gpu_lib
    from cupyx.scipy import ndimage as ndi_gpu

    gpu_img = cp.asarray(img_arr)
    gpu_lung = cp.asarray(lung_mask)

    soft_tissue = (gpu_img > -500) & (~gpu_lung)

    struct = ndi_gpu.generate_binary_structure(3, 1)
    eroded_tissue = ndi_gpu.binary_erosion(
        soft_tissue, structure=struct, iterations=ISOLATION_EROSION, brute_force=True
    )

    cpu_eroded = cp.asnumpy(eroded_tissue)
    labels, num = ndimage.label(cpu_eroded)

    if num == 0:
        return cp.asnumpy(soft_tissue)

    center_z, center_y, center_x = np.array(img_arr.shape) // 2
    best_label = 0
    min_dist = float('inf')

    objs = ndimage.find_objects(labels)
    sizes = ndimage.sum(cpu_eroded, labels, range(1, num + 1))
    sorted_indices = np.argsort(sizes)[::-1][:5]

    for idx in sorted_indices:
        if sizes[idx] < 2000: continue
        loc = objs[idx]
        z_c = (loc[0].start + loc[0].stop) // 2
        y_c = (loc[1].start + loc[1].stop) // 2
        x_c = (loc[2].start + loc[2].stop) // 2

        dist = (z_c - center_z) ** 2 + (y_c - center_y) ** 2 + (x_c - center_x) ** 2

        if dist < min_dist:
            min_dist = dist
            best_label = idx + 1

    if best_label == 0: best_label = np.argmax(sizes) + 1
    heart_core_cpu = (labels == best_label)

    gpu_heart_core = cp.asarray(heart_core_cpu)
    restored_mask = ndi_gpu.binary_dilation(
        gpu_heart_core, structure=struct, iterations=RESTORE_DILATION, brute_force=True
    )

    final_roi_gpu = restored_mask & (gpu_img > -500) & (~gpu_lung)
    return cp.asnumpy(final_roi_gpu)


# =========================================================================
#                             文件处理工具 (【修正部分】)
# =========================================================================

def get_bbox_with_padding(mask, padding, shape):
    slices = ndimage.find_objects(mask)
    if not slices: return None
    s = slices[0]
    z_min, z_max = max(0, s[0].start - padding), min(shape[0], s[0].stop + padding)
    y_min, y_max = max(0, s[1].start - padding), min(shape[1], s[1].stop + padding)
    x_min, x_max = max(0, s[2].start - padding), min(shape[2], s[2].stop + padding)
    return (z_min, z_max, y_min, y_max, x_min, x_max)


def save_cropped_nifti(arr, bbox, original_itk, save_path):
    z1, z2, y1, y2, x1, x2 = bbox
    cropped_arr = arr[z1:z2, y1:y2, x1:x2]
    out_itk = sitk.GetImageFromArray(cropped_arr)
    out_itk.SetSpacing(original_itk.GetSpacing())
    out_itk.SetDirection(original_itk.GetDirection())

    original_origin = original_itk.GetOrigin()
    original_idx = [int(x1), int(y1), int(z1)]
    new_origin = original_itk.TransformIndexToPhysicalPoint(original_idx)
    out_itk.SetOrigin(new_origin)

    # --- 【核心修正】安全保存策略 ---
    # 构造同目录下的临时文件名
    temp_path = save_path + ".temp.nii.gz"

    try:
        # 1. 写入临时文件 (ITK 只要看到 .nii.gz 就会正常工作)
        sitk.WriteImage(out_itk, temp_path)

        # 2. 如果目标文件已存在，先删除
        if os.path.exists(save_path):
            os.remove(save_path)

        # 3. 重命名为最终的复杂名称 (包含 .img.nii.gz)
        os.rename(temp_path, save_path)

    except Exception as e:
        # 如果重命名失败，打印错误并尝试清理
        print(f"Error saving {save_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_single_case(case_dir, case_id, cupy_lib=None):
    """处理单个病例"""
    out_dir = os.path.join(OUTPUT_ROOT, case_id)
    if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)

    # 目标文件名 (包含 .img.nii.gz)
    save_img_path = os.path.join(out_dir, f"{case_id}.img.nii.gz")
    save_lbl_path = os.path.join(out_dir, f"{case_id}.label.nii.gz")

    if os.path.exists(save_img_path):
        return None

    img_path = os.path.join(case_dir, f"{case_id}.img.nii.gz")
    lbl_path = os.path.join(case_dir, f"{case_id}.label.nii.gz")

    if not os.path.exists(img_path):
        img_path_alt = os.path.join(case_dir, "image.nii.gz")
        if os.path.exists(img_path_alt):
            img_path = img_path_alt
        else:
            return f"❌ [Skipped] {case_id}: No Image found"

    itk_img = sitk.ReadImage(img_path)
    itk_lbl = sitk.ReadImage(lbl_path) if os.path.exists(lbl_path) else None

    img_arr = sitk.GetArrayFromImage(itk_img)
    lbl_arr = sitk.GetArrayFromImage(itk_lbl) if itk_lbl else None

    # 1. CPU 肺分割
    lung_mask = get_lung_mask_cpu(img_arr)

    # 2. GPU 核心处理
    if cupy_lib:
        roi_mask = process_shrink_wrap_gpu(img_arr, lung_mask, cupy_lib)
    else:
        roi_mask = (img_arr > -500) & (~lung_mask)

    # 3. 统计
    loss_str = "无标签"
    if lbl_arr is not None:
        total_lbl = np.sum(lbl_arr > 0)
        missed_lbl = np.sum((lbl_arr > 0) & (~roi_mask))
        loss_rate = (missed_lbl / total_lbl * 100) if total_lbl > 0 else 0
        loss_str = f"{loss_rate:.2f}%"

    img_processed = img_arr.copy()
    img_processed[~roi_mask] = BG_VALUE

    bbox = get_bbox_with_padding(roi_mask, CROP_PADDING, img_arr.shape)
    if bbox is None: return f"❌ [Error] {case_id}: Empty ROI"

    # 调用修改后的保存函数
    save_cropped_nifti(img_processed, bbox, itk_img, save_img_path)
    if lbl_arr is not None:
        save_cropped_nifti(lbl_arr, bbox, itk_lbl, save_lbl_path)

    original_shape = str(img_arr.shape)
    cropped_shape = str((bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]))

    return f"✅ [GPU] {case_id} | 尺寸: {original_shape} -> {cropped_shape} | 损失: {loss_str}"


# =========================================================================
#                             多进程 Worker
# =========================================================================

def gpu_worker(gpu_id, case_dirs):
    """Worker 进程"""
    import cupy as cp

    try:
        cp.cuda.Device(gpu_id).use()
    except Exception as e:
        print(f"[GPU {gpu_id}] Init Error: {e}")
        cp = None

    pbar = tqdm(case_dirs, desc=f"[GPU {gpu_id}]", position=gpu_id + 1, leave=False)

    for case_dir in pbar:
        case_id = os.path.basename(case_dir)
        try:
            result_msg = process_single_case(case_dir, case_id, cupy_lib=cp)
            if result_msg:
                tqdm.write(f"[GPU-{gpu_id}] {result_msg}")
        except Exception as e:
            tqdm.write(f"❌ [GPU-{gpu_id}] Error processing {case_id}: {e}")


# =========================================================================
#                               主程序
# =========================================================================

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # 1. 获取所有病例
    all_case_dirs = sorted(glob.glob(os.path.join(INPUT_ROOT, "*")))
    target_case_dirs = []
    for d in all_case_dirs:
        if not os.path.isdir(d): continue
        case_id_str = os.path.basename(d)
        if case_id_str.isdigit():
            case_id_int = int(case_id_str)
            if START_ID <= case_id_int <= END_ID:
                target_case_dirs.append(d)

    # 2. 显卡分配
    try:
        import cupy
        num_gpus = cupy.cuda.runtime.getDeviceCount()
        available_devices = list(range(num_gpus))
        if VISIBLE_DEVICES:
            available_devices = [d for d in VISIBLE_DEVICES if d < num_gpus]
    except ImportError:
        print("未检测到 CuPy。")
        return

    num_workers = len(available_devices)
    if num_workers == 0:
        print("无可用 GPU。")
        return

    print(f"==================================================")
    print(f" 并行模式: {num_workers} 个 GPU Worker")
    print(f" 任务总数: {len(target_case_dirs)}")
    print(f"==================================================")

    chunk_size = math.ceil(len(target_case_dirs) / num_workers)
    chunks = [target_case_dirs[i:i + chunk_size] for i in range(0, len(target_case_dirs), chunk_size)]

    processes = []
    print("\n🚀 正在启动...\n")

    for i, gpu_id in enumerate(available_devices):
        if i < len(chunks):
            case_subset = chunks[i]
            p = mp.Process(target=gpu_worker, args=(gpu_id, case_subset))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print("\n" * (num_workers + 1))
    print("-" * 60)
    print("✅ 全部完成。")


if __name__ == "__main__":
    main()