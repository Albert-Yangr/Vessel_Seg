import os
import shutil

# 定义文件夹路径
train_dir = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-origin/train"
train45_dir = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-origin/train45"

# 安全开关：True 表示只打印不删除，False 表示真实删除
DRY_RUN = False


def remove_duplicate_folders():
    if not os.path.exists(train_dir) or not os.path.exists(train45_dir):
        print("错误：找不到指定的 train 或 train45 文件夹，请检查路径。")
        return

    # 获取 train45 中的所有子文件夹名称
    train45_subdirs = [f for f in os.listdir(train45_dir) if os.path.isdir(os.path.join(train45_dir, f))]

    count = 0
    for subdir in train45_subdirs:
        target_path = os.path.join(train_dir, subdir)

        # 如果这个子文件夹在 train 中也存在
        if os.path.exists(target_path) and os.path.isdir(target_path):
            if DRY_RUN:
                print(f"[试运行] 准备删除: {target_path}")
            else:
                print(f"[执行删除] 正在删除: {target_path}")
                shutil.rmtree(target_path)
            count += 1

    if DRY_RUN:
        print(f"\n试运行结束。共有 {count} 个文件夹匹配。如果确认无误，请将代码中的 DRY_RUN 改为 False 并重新运行。")
    else:
        print(f"\n清理完成！共删除了 {count} 个重复的子文件夹。")


if __name__ == "__main__":
    remove_duplicate_folders()