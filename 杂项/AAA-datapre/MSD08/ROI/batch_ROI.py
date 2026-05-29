import os
import shutil
import time
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage import morphology
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

# ================= 1. 科研配置区域 =================
# 输入输出路径
SRC_ROOT = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-61-clip/all"
DST_ROOT = "/home/yangrui/Project/Base-model/datasets/MSD08/MSD-61-ROI2/all"

# 肝脏 ROI 提取参数
PARAM_BODY_THRESH = -150  # 提取主体的 HU 阈值
PARAM_BRIDGE_BREAK = 3  # 微腐蚀半径 (断开粘连)
PARAM_RECOVER = 4  # 膨胀恢复半径

# 【新增】截断参数 (针对肝脏血管优化)
ENABLE_CLIPPING = True
CLIP_MIN = -200.0
CLIP_MAX = 400.0

COPY_LABELS = True


# =================================================

def extract_liver_refined_roi_with_clip(img_path, out_path, lab_path=None):
    """
    进阶方案：微腐蚀断桥 + 最大连通域锁定 + 膨胀恢复 + HU值截断
    """
    try:
        # 1. 读取数据
        img = nib.load(img_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # 2. 初始主体掩码 (用于确定 ROI 范围)
        binary_mask = data > PARAM_BODY_THRESH
        binary_mask = ndimage.binary_fill_holes(binary_mask)

        # 3. 微腐蚀断开细小粘连
        selem_break = morphology.ball(PARAM_BRIDGE_BREAK)
        eroded_mask = morphology.binary_erosion(binary_mask, selem_break)

        # 4. 二次锁定最大连通域
        labeled, num = ndimage.label(eroded_mask)
        if num == 0:
            return "Fail: Erosion too strong", 100.0

        region_sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
        refined_core = (labeled == (np.argmax(region_sizes) + 1))

        # 5. 膨胀恢复原始体积
        selem_recover = morphology.ball(PARAM_RECOVER)
        final_mask = morphology.binary_dilation(refined_core, selem_recover)
        final_mask = ndimage.binary_fill_holes(final_mask)

        # === 6. 标签完整性验证 ===
        damage_rate = 0.0
        integrity_msg = "[安全] 损伤: 0.00%"
        if lab_path and os.path.exists(lab_path):
            gt_data = nib.load(lab_path).get_fdata()
            gt_voxels = (gt_data > 0)
            gt_count = np.sum(gt_voxels)

            if gt_count > 0:
                preserved = np.logical_and(gt_voxels, final_mask)
                p_count = np.sum(preserved)
                loss = gt_count - p_count
                damage_rate = (loss / gt_count) * 100

                if loss > 0:
                    integrity_msg = f"[警告] 标签受损! 损伤率: {damage_rate:.4f}%"
                else:
                    integrity_msg = f"[安全] 损伤: 0.00% (保留: 100%)"

        # === 7. 【核心新增】截断与遮罩处理 ===
        # 先将图像转为 float32
        processed_data = data.astype(np.float32)

        # 执行截断操作
        if ENABLE_CLIPPING:
            processed_data = np.clip(processed_data, CLIP_MIN, CLIP_MAX)
            clip_msg = f"Clipped [{CLIP_MIN}, {CLIP_MAX}]"
        else:
            clip_msg = "No Clip"

        # 应用 ROI 遮罩：将 ROI 之外的区域设为背景值 (通常设为截断下限或空气值)
        # 这里建议设为 -1024 (标准空气值) 或 CLIP_MIN (-200)
        # 为了保持你之前的习惯，我们设为 -1024
        processed_data[~final_mask] = -100

        # 8. 保存结果
        new_img = nib.Nifti1Image(processed_data, affine, header)
        nib.save(new_img, out_path)
        return f"Success | {clip_msg} | {integrity_msg}", damage_rate

    except Exception as e:
        return f"Error: {str(e)}", 100.0


def process_case(case_id):
    src_case_dir = os.path.join(SRC_ROOT, case_id)
    dst_case_dir = os.path.join(DST_ROOT, case_id)
    os.makedirs(dst_case_dir, exist_ok=True)

    img_name = f"{case_id}.img.nii.gz"
    lab_name = f"{case_id}.label.nii.gz"

    src_img = os.path.join(src_case_dir, img_name)
    src_lab = os.path.join(src_case_dir, lab_path := os.path.join(src_case_dir, lab_name))
    dst_img = os.path.join(dst_case_dir, img_name)
    dst_lab = os.path.join(dst_case_dir, lab_name)

    # 真正的处理逻辑
    status, damage = extract_liver_refined_roi_with_clip(src_img, dst_img,
                                                         lab_path=src_lab if os.path.exists(src_lab) else None)

    # 复制标签
    if COPY_LABELS and os.path.exists(src_lab):
        shutil.copy(src_lab, dst_lab)

    return f"Case {case_id}: {status}"


def main():
    print("=== 开始肝脏血管【ROI提取 + HU截断】流水线 ===")
    case_ids = sorted([d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))])

    start_time = time.time()
    results = []

    # 使用多进程加速
    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in tqdm(executor.map(process_case, case_ids), total=len(case_ids), desc="正在处理"):
            results.append(result)
            tqdm.write(result)

    print("\n" + "=" * 40)
    print(f"处理完成! 总耗时: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()