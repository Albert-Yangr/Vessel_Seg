import os
import shutil
import glob

# 源文件夹路径
source_folder = '/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-refile/test'
# 目标文件夹路径
target_folder = '/home/yangrui/Project/Base-model/local_results/output/CAS2023/CAS2023-refile/GT'

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)
print(f"目标文件夹已创建: {target_folder}")

# 获取所有子文件夹
subfolders = [f for f in os.listdir(source_folder)
              if os.path.isdir(os.path.join(source_folder, f))]
print(f"找到 {len(subfolders)} 个子文件夹")

# 提取所有label文件
extracted_count = 0
for folder in subfolders:
    folder_path = os.path.join(source_folder, folder)

    # 查找该文件夹中的所有.label.nii.gz文件
    label_files = glob.glob(os.path.join(folder_path, "*.label.nii.gz"))

    for label_file in label_files:
        # 获取文件名
        filename = os.path.basename(label_file)
        # 目标文件路径
        target_path = os.path.join(target_folder, filename)

        # 复制文件
        shutil.copy2(label_file, target_path)
        extracted_count += 1
        print(f"已提取: {filename}")

print(f"\n完成! 共提取了 {extracted_count} 个label文件")
print(f"文件保存在: {target_folder}")

# 显示前几个文件确认
print("\n提取的文件列表（前10个）:")
extracted_files = os.listdir(target_folder)
for i, file in enumerate(extracted_files[:10]):
    print(f"  {i + 1}. {file}")
if len(extracted_files) > 10:
    print(f"  ... 还有 {len(extracted_files) - 10} 个文件")