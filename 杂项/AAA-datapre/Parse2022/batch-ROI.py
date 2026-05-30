import os
import numpy as np
import nibabel as nib
from skimage import measure, morphology
from scipy import ndimage
import glob
import logging
from tqdm import tqdm

# 配置日志：将日志同时输出到文件，方便你跑完后查看统计
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("roi_integrity_check.log"),  # 所有的警告都会保存在这个文件里
        logging.StreamHandler()
    ]
)


# ==========================================
# 1. 核心 Mask 生成逻辑 (保持不变)
# ==========================================

def get_body_mask(img_data):
    binary = img_data < -300
    labels = measure.label(binary, connectivity=1)

    d, h, w = img_data.shape
    corners = [
        (0, 0, 0), (d - 1, 0, 0), (0, h - 1, 0), (d - 1, h - 1, 0),
        (0, 0, w - 1), (d - 1, 0, w - 1), (0, h - 1, w - 1), (d - 1, h - 1, w - 1)
    ]
    background_labels = set()
    for c in corners:
        bg_lbl = labels[c]
        if bg_lbl != 0:
            background_labels.add(bg_lbl)
    for bg_lbl in background_labels:
        labels[labels == bg_lbl] = 0

    props = measure.regionprops(labels)
    props.sort(key=lambda x: x.area, reverse=True)

    lung_mask = np.zeros_like(binary, dtype=np.uint8)
    if not props:
        return lung_mask

    lung_mask[labels == props[0].label] = 1
    if len(props) > 1:
        if props[1].area > (props[0].area * 0.2):
            lung_mask[labels == props[1].label] = 1

    return lung_mask


def get_smooth_convex_mask(lung_mask, sigma=3):
    hull_mask = np.zeros_like(lung_mask)
    slices_num = lung_mask.shape[2]

    for i in range(slices_num):
        slice_img = lung_mask[:, :, i]
        if np.sum(slice_img) > 0:
            try:
                hull = morphology.convex_hull_image(slice_img)
                hull_mask[:, :, i] = hull
            except Exception:
                hull_mask[:, :, i] = slice_img

    float_mask = hull_mask.astype(float)
    smoothed_float = ndimage.gaussian_filter(float_mask, sigma=sigma)
    final_mask = (smoothed_float > 0.5).astype(np.uint8)

    return final_mask


def generate_roi_mask(img_data):
    lung_mask = get_body_mask(img_data)
    if np.sum(lung_mask) == 0:
        return None
    # 保持 Sigma=4，追求平滑
    smooth_mask = get_smooth_convex_mask(lung_mask, sigma=4)
    return smooth_mask


# ==========================================
# 2. 【核心修改】仅检测，不修补
# ==========================================

def check_mask_coverage(mask, lbl_data, pid="Unknown"):
    """
    检查 Mask 完整性。
    只报警，不修改 mask。
    """
    if lbl_data is None:
        return

    # 找出 Label 存在 但 Mask 缺失 的区域
    missed_pixels = (lbl_data > 0) & (mask == 0)
    miss_count = np.sum(missed_pixels)
    total_label = np.sum(lbl_data > 0)

    if miss_count > 0:
        ratio = (miss_count / total_label) * 100 if total_label > 0 else 0
        # 记录到日志文件，方便后续筛选坏数据
        logging.warning(f"【损失检测】ID {pid}: 丢失 {miss_count} 像素 ({ratio:.4f}%) -> 真实血管在 Mask 外！")
    else:
        # 如果你想看完美的例子，可以取消注释下面这行
        # logging.info(f"ID {pid}: 完美覆盖。")
        pass


# ==========================================
# 3. 核心裁剪逻辑
# ==========================================

def get_bbox_from_mask(mask, margin=0):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None, None, None

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1

    min_coords = np.maximum(min_coords - margin, 0)
    max_coords = np.minimum(max_coords + margin, mask.shape)

    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return slices, min_coords, max_coords


def crop_and_save(img_path, lbl_path, output_dir):
    try:
        pid = os.path.basename(os.path.dirname(img_path))

        # 1. 读取
        img_obj = nib.load(img_path)
        img_data = img_obj.get_fdata()
        original_affine = img_obj.affine

        lbl_data = None
        if lbl_path and os.path.exists(lbl_path):
            lbl_obj = nib.load(lbl_path)
            lbl_data = lbl_obj.get_fdata()

        # 2. 生成 ROI Mask
        mask = generate_roi_mask(img_data)
        if mask is None:
            logging.warning(f"跳过: {img_path} 无法生成有效Mask")
            return

        # =================================================
        # 【修改点】仅检测完整性，不做任何修改
        # =================================================
        check_mask_coverage(mask, lbl_data, pid)

        # 3. 计算裁剪坐标
        crop_slices, min_coords, _ = get_bbox_from_mask(mask, margin=0)
        if crop_slices is None:
            logging.warning(f"跳过: {img_path} Mask为空")
            return

        # 4. 执行裁剪
        cropped_img_data = img_data[crop_slices].copy()
        cropped_mask = mask[crop_slices]

        # 5. 应用 Mask (背景置黑)
        # 注意：既然我们选择了“不修复”，那么这里置黑操作
        # 确实会把位于 Mask 外面的血管像素变成背景值（被切掉）。
        # 这是预期的“按算法裁剪”行为。
        min_val = np.min(img_data)
        cropped_img_data[cropped_mask == 0] = min_val

        # 6. 保存 Image
        new_affine = original_affine.copy()
        translation = np.dot(original_affine[:3, :3], min_coords)
        new_affine[:3, 3] += translation

        filename_img = os.path.basename(img_path)
        save_path_img = os.path.join(output_dir, filename_img)
        new_img_obj = nib.Nifti1Image(cropped_img_data, new_affine, img_obj.header)
        nib.save(new_img_obj, save_path_img)

        # 7. 保存 Label
        if lbl_data is not None:
            cropped_lbl_data = lbl_data[crop_slices]

            # 【重要】为了保证 Image 和 Label 严格对齐
            # 即使 Label 被切了，Label 文件里也必须体现这个“被切”的状态
            # 否则训练时 Input 和 Label 不一致（Input 是黑的，Label 却是 1）会产生严重的噪声

            # 方案 A: 忠实记录裁剪结果 (Label 也应用 Mask) -> 推荐用于训练集制作
            # cropped_lbl_data[cropped_mask == 0] = 0

            # 方案 B: 保留原始 Label (Label 不应用 Mask)
            # 这样你可以看到 Mask 外面还有 Label，方便可视化检查
            # 但拿去训练的话，网络会很困惑：为什么背景全黑的地方还有 Label？
            # 既然你主要为了检测，这里先选 B，不做额外破坏，只做物理裁剪

            filename_lbl = os.path.basename(lbl_path)
            save_path_lbl = os.path.join(output_dir, filename_lbl)
            new_lbl_obj = nib.Nifti1Image(cropped_lbl_data.astype(np.uint8), new_affine, lbl_obj.header)
            nib.save(new_lbl_obj, save_path_lbl)

    except Exception as e:
        logging.error(f"处理失败 {img_path}: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
# 4. 主流程
# ==========================================

def main():
    # ==== 配置路径 ====
    input_root = "/home/yangrui/Project/Base-model/datasets/Parse/Parse-reshape"
    output_root = "/home/yangrui/Project/Base-model/datasets/Parse/Parse-ROI"
    splits = ["all"]
    # ================

    for split in splits:
        src_split_path = os.path.join(input_root, split)
        dst_split_path = os.path.join(output_root, split)

        if not os.path.exists(src_split_path):
            continue

        patient_ids = [d for d in os.listdir(src_split_path) if os.path.isdir(os.path.join(src_split_path, d))]
        print(f"正在处理 {split} 集，共 {len(patient_ids)} 个病例...")

        for pid in tqdm(patient_ids, desc="Processing"):
            patient_src_dir = os.path.join(src_split_path, pid)
            patient_dst_dir = os.path.join(dst_split_path, pid)

            if not os.path.exists(patient_dst_dir):
                os.makedirs(patient_dst_dir)

            img_files = glob.glob(os.path.join(patient_src_dir, "*img.nii.gz")) + \
                        glob.glob(os.path.join(patient_src_dir, "image.nii.gz"))
            lbl_files = glob.glob(os.path.join(patient_src_dir, "*label.nii.gz")) + \
                        glob.glob(os.path.join(patient_src_dir, "label.nii.gz"))

            if not img_files:
                continue

            img_path = img_files[0]
            lbl_path = lbl_files[0] if lbl_files else None

            crop_and_save(img_path, lbl_path, patient_dst_dir)

    print(f"\n全部完成！检测日志已生成在当前目录的 roi_integrity_check.log")


if __name__ == "__main__":
    main()