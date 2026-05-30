import numpy as np
import nibabel as nib
from skimage import morphology, measure
from scipy import ndimage
import os
import argparse
import time


def extract_heart_advanced(input_path, output_path,
                           threshold_hu=-300,
                           erosion_radius=8,
                           dilation_radius=20):
    """
    高级心脏提取：通过核心腐蚀和受控生长，去除粘连的粗壮血管。

    参数原理:
    1. threshold_hu: 基础阈值。
    2. erosion_radius (腐蚀半径): 关键参数！越大，“断桥”能力越强，能把粘连断得越干净。建议 6-10。
    3. dilation_radius (膨胀半径): 关键参数！决定了以心脏核心为基础，向外保留多厚的区域。
       必须 > erosion_radius。数值越大，保留的冠脉越多，但如果太大，会把无关血管包进来。
       建议设置为: erosion_radius + 10 到 15 (即 18-25左右)。
    """
    print(f"正在处理: {input_path}")
    start_time = time.time()

    # 1. 读取数据
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine

    # 2. 基础阈值掩码 (包含心脏、冠脉、粘连血管)
    print(f"应用阈值 > {threshold_hu}...")
    binary_mask = data > threshold_hu

    # 3. 核心分离 (Deep Erosion)
    # 目的：断开细小的连接桥，消除冠脉，缩小粗血管，只保留厚实的心室核心
    print(f"正在剥离粘连 (腐蚀半径 {erosion_radius})...")
    selem_erode = morphology.ball(erosion_radius)
    eroded_mask = morphology.binary_erosion(binary_mask, selem_erode)

    # 4. 提取最大连通域 (只取心脏核心)
    print("提取心脏核心...")
    labeled_mask, num_features = ndimage.label(eroded_mask)

    if num_features == 0:
        print("错误：腐蚀后没有剩余区域，请降低 erosion_radius 或 降低 threshold_hu")
        return

    # 计算各区域大小并排序
    region_sizes = [(i, np.sum(labeled_mask == i)) for i in range(1, num_features + 1)]
    region_sizes.sort(key=lambda x: x[1], reverse=True)

    largest_label = region_sizes[0][0]
    heart_core = (labeled_mask == largest_label)
    print(f"找到心脏核心，体积: {region_sizes[0][1]} 体素")

    # 5. 建立安全区 (Geofencing / Controlled Dilation)
    # 目的：以纯净核心为种子，向外扩张一个安全距离。
    # 这个距离足以包住冠状动脉，但够不到那个被断开的粗血管。
    print(f"生成安全区 (膨胀半径 {dilation_radius})...")

    # 这里使用距离变换代替形态学膨胀，速度更快且边缘平滑
    # 计算全图距离核心的距离
    dist_map = ndimage.distance_transform_edt(~heart_core)
    # 只要距离小于 dilation_radius 的区域都在安全区内
    safety_zone = dist_map <= dilation_radius

    # 6. 最终切割 (Final Cut)
    # 逻辑：保留 (原始阈值内) 且 (位于安全区内) 的像素
    # 这样既保留了冠脉的细节（由binary_mask决定），又排除了远处的粘连血管（由safety_zone决定）
    final_mask = np.logical_and(binary_mask, safety_zone)

    # 7. (可选) 闭运算修补
    # 修复可能存在的微小断裂
    final_mask = morphology.binary_closing(final_mask, morphology.ball(2))

    # 8. 保存结果
    # 背景设为 -1024 (空气)
    masked_data = data.copy()
    masked_data[~final_mask] = -1000

    new_img = nib.Nifti1Image(masked_data, affine, img.header)
    nib.save(new_img, output_path)

    print(f"处理完成，耗时 {time.time() - start_time:.2f}s")
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='高级心脏区域提取')
    parser.add_argument('--input', '-i', type=str,
                        default="/home/yangrui/Project/Base-model/datasets/AAA-datapre/ROI/imageCAS/data/img/971.img.nii.gz",
                        help='输入文件路径')
    parser.add_argument('--output', '-o', type=str,
                        default="/home/yangrui/Project/Base-model/datasets/AAA-datapre/ROI/imageCAS/data/output/971_ROI.nii.gz",
                        help='输出文件路径')


    # 增加参数调整接口
    parser.add_argument('--erode', type=int,
                        default=12, help='腐蚀半径 (用于断开粘连)')
    parser.add_argument('--dilate', type=int,
                        default=14, help='膨胀半径 (用于包裹冠脉)')

    args = parser.parse_args()

    extract_heart_advanced(args.input, args.output,
                           erosion_radius=args.erode,
                           dilation_radius=args.dilate)
