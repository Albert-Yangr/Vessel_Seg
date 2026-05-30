import os
import numpy as np
import nibabel as nib
from pathlib import Path
import re


class BatchNiftiCropper:
    def __init__(self, margin=(10, 10, 10)):
        self.margin = margin

    def find_bounding_box(self, mask_data):
        """找到包含标签的最小边界框"""
        nonzero_indices = np.where(mask_data > 0)

        if len(nonzero_indices[0]) == 0:
            raise ValueError("标签图像中没有找到任何非零像素")

        min_x, max_x = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
        min_y, max_y = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
        min_z, max_z = np.min(nonzero_indices[2]), np.max(nonzero_indices[2])

        return (min_x, max_x, min_y, max_y, min_z, max_z)

    def calculate_crop_indices(self, bbox, image_shape):
        """计算裁剪索引"""
        min_x, max_x, min_y, max_y, min_z, max_z = bbox

        crop_min_x = max(0, min_x - self.margin[0])
        crop_max_x = min(image_shape[0], max_x + self.margin[0] + 1)
        crop_min_y = max(0, min_y - self.margin[1])
        crop_max_y = min(image_shape[1], max_y + self.margin[1] + 1)
        crop_min_z = max(0, min_z - self.margin[2])
        crop_max_z = min(image_shape[2], max_z + self.margin[2] + 1)

        return (crop_min_x, crop_max_x, crop_min_y, crop_max_y, crop_min_z, crop_max_z)

    def update_affine_matrix(self, original_affine, crop_min_x, crop_min_y, crop_min_z):
        """更新仿射矩阵以保持空间坐标"""
        voxel_sizes = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))
        new_affine = original_affine.copy()
        new_affine[:3, 3] = original_affine[:3, 3] + [
            crop_min_x * voxel_sizes[0],
            crop_min_y * voxel_sizes[1],
            crop_min_z * voxel_sizes[2]
        ]
        return new_affine

    def crop_single_case(self, image_path, mask_path, output_image_path, output_mask_path):
        """处理单个病例"""
        try:
            # 加载数据
            image_img = nib.load(image_path)
            mask_img = nib.load(mask_path)
            image_data = image_img.get_fdata()
            mask_data = mask_img.get_fdata()

            # 验证尺寸匹配
            if image_data.shape != mask_data.shape:
                print(f"尺寸不匹配: {image_path}")
                return False

            # 找到边界框并计算裁剪范围
            bbox = self.find_bounding_box(mask_data)
            crop_indices = self.calculate_crop_indices(bbox, image_data.shape)
            crop_min_x, crop_max_x, crop_min_y, crop_max_y, crop_min_z, crop_max_z = crop_indices

            # 执行裁剪
            cropped_image = image_data[crop_min_x:crop_max_x, crop_min_y:crop_max_y, crop_min_z:crop_max_z]
            cropped_mask = mask_data[crop_min_x:crop_max_x, crop_min_y:crop_max_y, crop_min_z:crop_max_z]

            # 更新仿射矩阵
            new_affine = self.update_affine_matrix(image_img.affine, crop_min_x, crop_min_y, crop_min_z)

            # 创建并保存新图像
            cropped_image_img = nib.Nifti1Image(cropped_image, new_affine)
            cropped_mask_img = nib.Nifti1Image(cropped_mask, new_affine)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

            nib.save(cropped_image_img, output_image_path)
            nib.save(cropped_mask_img, output_mask_path)

            # 计算体积减少比例
            original_volume = np.prod(image_data.shape)
            cropped_volume = np.prod(cropped_image.shape)
            reduction_ratio = 1 - (cropped_volume / original_volume)

            print(f"✓ 成功处理: {Path(image_path).name}")
            print(f"  尺寸: {image_data.shape} → {cropped_image.shape}")
            print(f"  体积减少: {reduction_ratio:.1%}")

            return True

        except Exception as e:
            print(f"✗ 处理失败: {Path(image_path).name} - 错误: {e}")
            return False

    def find_case_folders(self, base_dir, start=1, end=200):
        """查找病例文件夹"""
        case_folders = []
        for i in range(start, end + 1):
            case_path = os.path.join(base_dir, str(i))
            if os.path.exists(case_path):
                case_folders.append((i, case_path))
        return case_folders

    def process_batch(self, input_base_dir, output_base_dir, start_case=900, end_case=999):
        """批量处理所有病例"""
        print(f"开始处理病例 {start_case} 到 {end_case}...")
        print(f"输入目录: {input_base_dir}")
        print(f"输出目录: {output_base_dir}")
        print("-" * 60)

        # 查找所有病例文件夹
        case_folders = self.find_case_folders(input_base_dir, start_case, end_case)

        if not case_folders:
            print("未找到任何病例文件夹！")
            return

        print(f"找到 {len(case_folders)} 个病例文件夹")

        success_count = 0
        total_cases = len(case_folders)

        for case_num, case_dir in case_folders:
            # 构建文件路径
            image_path = os.path.join(case_dir, f"{case_num}.img.nii.gz")
            mask_path = os.path.join(case_dir, f"{case_num}.label.nii.gz")

            # 检查文件是否存在
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print(f"✗ 病例 {case_num}: 文件不存在")
                continue

            # 构建输出路径（保持相同的目录结构）
            output_case_dir = os.path.join(output_base_dir, str(case_num))
            output_image_path = os.path.join(output_case_dir, f"{case_num}.img.nii.gz")
            output_mask_path = os.path.join(output_case_dir, f"{case_num}.label.nii.gz")

            # 处理单个病例
            if self.crop_single_case(image_path, mask_path, output_image_path, output_mask_path):
                success_count += 1

        # 输出统计信息
        print("-" * 60)
        print(f"处理完成！")
        print(f"成功处理: {success_count}/{total_cases} 个病例")
        print(f"成功率: {success_count / total_cases * 100:.1f}%")

    def process_specific_cases(self, input_base_dir, output_base_dir, case_numbers):
        """处理指定的病例编号"""
        print(f"处理指定病例: {case_numbers}")

        success_count = 0
        for case_num in case_numbers:
            case_dir = os.path.join(input_base_dir, str(case_num))
            image_path = os.path.join(case_dir, f"{case_num}.img.nii.gz")
            mask_path = os.path.join(case_dir, f"{case_num}.label.nii.gz")

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print(f"✗ 病例 {case_num}: 文件不存在")
                continue

            output_case_dir = os.path.join(output_base_dir, str(case_num))
            output_image_path = os.path.join(output_case_dir, f"{case_num}.img.nii.gz")
            output_mask_path = os.path.join(output_case_dir, f"{case_num}.label.nii.gz")

            if self.crop_single_case(image_path, mask_path, output_image_path, output_mask_path):
                success_count += 1

        print(f"指定病例处理完成: {success_count}/{len(case_numbers)} 成功")


def main():
    # 配置路径
    INPUT_BASE_DIR = "/home/yangrui/Project/Base-model/datasets/imageCAS/ImageCAS-original/all/1-200"
    OUTPUT_BASE_DIR = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-clip-0/all"

    # 创建裁剪器实例
    cropper = BatchNiftiCropper(margin=(0, 0, 0))  # 可根据需要调整边缘大小

    # 方法1: 处理所有病例
    cropper.process_batch(INPUT_BASE_DIR, OUTPUT_BASE_DIR, 1, 2)




if __name__ == "__main__":
    main()