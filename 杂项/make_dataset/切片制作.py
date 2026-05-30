import os
import glob
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
# 如果没有安装 tqdm，可以 pip install tqdm，用来显示美观的进度条
from tqdm import tqdm

# 1. 设置数据集的基础路径
base_dir = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-origin/train"


def process_single_label(label_path, subdir_name):
    """
    处理单个文件的核心逻辑（独立出来以便多进程调用）
    """
    try:
        # 获取文件名并提取数字标号前缀
        filename = os.path.basename(label_path)
        prefix = filename.replace(".label.nii.gz", "")

        # 构建输出文件的绝对路径
        output_filename = f"{prefix}.slice.nii.gz"
        # 注意：这里直接放在原文件同级目录
        output_path = os.path.join(os.path.dirname(label_path), output_filename)

        # 如果已经存在，可以选择跳过以节省时间 (支持断点续传)
        if os.path.exists(output_path):
            return True, f"已存在，跳过: {output_filename}"

        # 3. 读取原始 NIfTI 标签文件
        nii_img = nib.load(label_path)
        data = np.asanyarray(nii_img.dataobj)

        out_dtype = data.dtype
        if out_dtype.itemsize == 1 and out_dtype.kind == 'i':
            out_dtype = np.uint8
        elif out_dtype == bool:
            out_dtype = np.uint8

        # 4. 创建一个全为 255 的新数组
        new_data = np.full(data.shape, 255, dtype=out_dtype)

        # 5. 获取几何中心
        cx = data.shape[0] // 2
        cy = data.shape[1] // 2
        cz = data.shape[2] // 2

        # 6. 提取十字切片
        new_data[cx, :, :] = data[cx, :, :]
        new_data[:, cy, :] = data[:, cy, :]
        new_data[:, :, cz] = data[:, :, cz]

        # 7. 保存文件
        new_img = nib.Nifti1Image(new_data, nii_img.affine, nii_img.header)
        new_img.set_data_dtype(out_dtype)
        nib.save(new_img, output_path)

        return True, f"成功生成: {output_filename}"

    except Exception as e:
        return False, f"处理文件 {label_path} 时发生错误: {e}"


def process_labels_multiprocess():
    if not os.path.exists(base_dir):
        print(f"❌ 错误：找不到路径 {base_dir}")
        return

    # 收集所有需要处理的文件任务
    tasks = []
    print("🔍 正在扫描文件...")
    for subdir_name in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        label_files = glob.glob(os.path.join(subdir_path, "*.label.nii.gz"))
        for label_path in label_files:
            tasks.append((label_path, subdir_name))

    total_files = len(tasks)
    print(f"🎯 共找到 {total_files} 个标签文件，准备开启多进程加速处理...")

    # 开启进程池，最大进程数设为 CPU 核心数 - 2 (保留一点资源给系统)
    max_workers = max(1, os.cpu_count() - 2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_label, task[0], task[1]) for task in tasks]

        # 使用 tqdm 监控进度
        for future in tqdm(as_completed(futures), total=total_files, desc="制作切片中"):
            success, msg = future.result()
            # 如果遇到错误，打印出来；成功的就不打印了，以免终端刷屏
            if not success:
                print(f"\n{msg}")

    print("-" * 40)
    print("🎉 所有标签文件处理完毕！")


if __name__ == "__main__":
    process_labels_multiprocess()