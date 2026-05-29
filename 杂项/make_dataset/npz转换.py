import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import traceback

warnings.filterwarnings("ignore")

# 🌟 配置你的数据集根目录
DATASET_DIR = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-origin/train-all"


def convert_single_file(nii_path):
    """
    转换单个 .nii.gz 文件为 .npy，包含完整的错误处理
    """
    nii_path = Path(nii_path)

    # 构造输出路径
    npy_name = nii_path.name.replace(".nii.gz", ".npy")
    npy_path = nii_path.parent / npy_name

    try:
        # 1. 检查文件是否为空
        if nii_path.stat().st_size == 0:
            print(f"\n⚠️ 文件为空: {nii_path}")
            return False

        # 2. 读取 NIfTI 文件
        itk_img = sitk.ReadImage(str(nii_path))
        arr = sitk.GetArrayFromImage(itk_img)

        # 3. 检查数组是否为空
        if arr.size == 0:
            print(f"\n⚠️ 数组为空: {nii_path}")
            return False

        # 4. 类型优化
        filename_lower = npy_name.lower()
        if "label" in filename_lower or "mask" in filename_lower or "seg" in filename_lower:
            arr = arr.astype(np.uint8)
        elif "img" in filename_lower or "image" in filename_lower:
            # 归一化处理，避免数值过大
            arr = arr.astype(np.float32)
            if arr.max() > 0:
                arr = arr / arr.max()  # 简单归一化
        else:
            # 默认使用 float32
            arr = arr.astype(np.float32)

        # 5. 保存文件
        np.save(str(npy_path), arr)
        return True

    except sitk.RuntimeError as e:
        # SimpleITK 读取错误（通常是文件损坏）
        print(f"\n❌ 读取失败 (SimpleITK): {nii_path.name} | {str(e)[:100]}")
        return False
    except MemoryError:
        print(f"\n❌ 内存不足: {nii_path.name} (文件过大)")
        return False
    except Exception as e:
        print(f"\n❌ 转换失败: {nii_path.name} | 错误: {str(e)[:100]}")
        return False


def convert_with_single_process(files):
    """单进程模式（降级方案）"""
    print("🔄 切换到单进程模式...")
    success_count = 0
    for nii_path in tqdm(files, desc="🔄 转换进度"):
        if convert_single_file(nii_path):
            success_count += 1
    return success_count


def main():
    root_path = Path(DATASET_DIR)
    if not root_path.exists():
        print(f"路径不存在: {root_path}")
        return

    print("🔍 正在扫描所有的 .nii.gz 文件...")
    all_nii_files = list(root_path.rglob("*.nii.gz"))

    if not all_nii_files:
        print("没有找到任何 .nii.gz 文件！")
        return

    print(f"📦 共找到 {len(all_nii_files)} 个文件等待转换。")

    # 检查文件大小分布
    sizes = [f.stat().st_size / (1024 ** 2) for f in all_nii_files]  # MB
    print(f"📊 文件大小统计: 最小={min(sizes):.2f}MB, 最大={max(sizes):.2f}MB, 平均={sum(sizes) / len(sizes):.2f}MB")

    # 使用更保守的多进程设置
    cpu_count = os.cpu_count()
    max_workers = max(1, min(4, cpu_count - 1))  # 最多使用4个进程，避免内存爆炸
    print(f"🚀 启动转换引擎 (使用 {max_workers} 个进程)...")

    success_count = 0
    failed_files = []

    # 添加启动方法设置（Linux环境）
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 分批提交，避免一次性提交太多任务
            batch_size = 100
            futures = {}

            for i in range(0, len(all_nii_files), batch_size):
                batch = all_nii_files[i:i + batch_size]
                for path in batch:
                    future = executor.submit(convert_single_file, path)
                    futures[future] = path

                # 处理当前批次的结果
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"🔄 批次 {i // batch_size + 1}/{(len(all_nii_files) - 1) // batch_size + 1}"):
                    path = futures[future]
                    try:
                        if future.result(timeout=30):  # 30秒超时
                            success_count += 1
                        else:
                            failed_files.append(path)
                    except Exception as e:
                        print(f"\n⚠️ 处理文件时出错: {path.name} | {str(e)[:100]}")
                        failed_files.append(path)

                futures.clear()  # 清空当前批次，释放内存

    except Exception as e:
        print(f"\n⚠️ 多进程池出现错误: {e}")
        print("切换到单进程模式继续处理...")
        success_count = convert_with_single_process(all_nii_files)

    # 输出结果
    print("\n" + "=" * 50)
    print(f"🎉 转换完成！")
    print(f"✅ 成功: {success_count} / {len(all_nii_files)}")
    print(f"❌ 失败: {len(failed_files)}")

    if failed_files:
        print("\n失败的文件列表（前10个）:")
        for f in failed_files[:10]:
            print(f"  - {f.name}")

        # 保存失败列表到文件
        with open("failed_conversion.txt", "w") as f:
            for path in failed_files:
                f.write(str(path) + "\n")
        print(f"\n💾 完整失败列表已保存到: failed_conversion.txt")

    print("=" * 50)


if __name__ == "__main__":
    main()