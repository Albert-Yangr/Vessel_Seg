import os
from pathlib import Path
from tqdm import tqdm

# 🌟 配置你要清理的数据集根目录
DATASET_DIR = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-origin/train-all"


def delete_npy_files(directory):
    root_path = Path(directory)
    if not root_path.exists():
        print(f"❌ 路径不存在: {root_path}")
        return

    print("🔍 正在扫描所有的 .npy 文件...")
    # rglob 会递归查找所有子文件夹中的 .npy 文件
    all_npy_files = list(root_path.rglob("*.npy"))

    if not all_npy_files:
        print("✅ 没有找到任何 .npy 文件，无需清理！")
        return

    print(f"⚠️ 共找到 {len(all_npy_files)} 个 .npy 文件等待删除。")

    # ---------------------------------------------------------
    # 安全锁：防止意外执行导致数据丢失，要求用户手动确认
    # ---------------------------------------------------------
    confirm = input("❓ 是否确认永久删除这些文件？该操作不可逆！(输入 y 确认, 其它任意键取消): ")
    if confirm.lower() != 'y':
        print("🚫 已取消删除。")
        return

    print("🗑️ 开始删除...")
    success_count = 0
    failed_files = []

    for npy_path in tqdm(all_npy_files, desc="🗑️ 删除进度"):
        try:
            # unlink() 用于删除文件
            npy_path.unlink()
            success_count += 1
        except Exception as e:
            failed_files.append((npy_path, str(e)))

    # 输出结果报告
    print("\n" + "=" * 50)
    print(f"🎉 清理完成！")
    print(f"✅ 成功删除: {success_count} / {len(all_npy_files)}")

    if failed_files:
        print(f"❌ 删除失败: {len(failed_files)} 个文件")
        print("失败文件列表（前10个）:")
        for f, err in failed_files[:10]:
            print(f"  - {f.name} | 错误: {err}")
    print("=" * 50)


if __name__ == "__main__":
    delete_npy_files(DATASET_DIR)