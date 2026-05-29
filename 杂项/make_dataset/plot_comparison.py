import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import numpy as np


def plot_multiple_3d_fine(file_paths, titles, save_path, elev=30, azim=-60):
    """
    动态多图并排渲染 3D 血管
    :param file_paths: NIfTI 文件路径列表 (支持任意数量)
    :param titles: 对应的标题列表
    :param save_path: 图片保存路径
    :param elev: 俯仰角 (上下旋转视角，范围 -90 到 90)
    :param azim: 方位角 (左右旋转视角，范围 -180 到 180)
    """
    num_plots = len(file_paths)

    # 动态调整画布尺寸，保证任意数量的图都不会拥挤，设置纯白背景
    fig = plt.figure(figsize=(8 * num_plots, 8), dpi=300)
    fig.patch.set_facecolor('white')  # 强制画布背景为纯白
    threshold = 0

    for idx, (file_path, title) in enumerate(zip(file_paths, titles)):
        print(f"正在渲染 ({idx + 1}/{num_plots}): {title}...")

        # 读取 nii.gz 数据
        nii_img = nib.load(file_path)
        target = nii_img.get_fdata()

        # 坐标系转置逻辑
        t = target.transpose(2, 1, 0)

        # 提取 3D 表面
        t_verts, t_faces, t_normals, t_values = measure.marching_cubes(t, threshold)

        # 创建子图，背景强制设为纯白
        ax = fig.add_subplot(1, num_plots, idx + 1, projection='3d小框')
        ax.set_facecolor('white')

        # 遵循老师的透明度和颜色设置 (alpha=0.2)
        t_mesh = Poly3DCollection(t_verts[t_faces], alpha=0.4)
        t_mesh.set_facecolor('red')
        ax.add_collection3d(t_mesh)

        # 边界设置
        ax.set_xlim(0, t.shape[0])
        ax.set_ylim(0, t.shape[1])
        ax.set_zlim(0, t.shape[2])

        # 保持真实物理比例，防止拓扑结构变形
        ax.set_box_aspect((t.shape[0], t.shape[1], t.shape[2]))

        # 关闭所有坐标轴、网格
        ax.axis('off')

        # 设置标题并调整距离，字体颜色设为黑色以防在白底上看不见
        ax.set_title(title, fontsize=24, pad=10, color='black')

        # 【核心新增】：动态调整视角
        ax.view_init(elev=elev, azim=azim)

    # 紧凑布局
    plt.tight_layout()

    # 【核心修复】：transparent=False 配合 facecolor='white' 确保输出纯白背景
    fig.savefig(save_path, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()
    print(f"✅ 精细渲染对比图已保存至: {save_path}")


# ================= 运行配置 (支持任意数量输入) =================

# 你可以随意增删列表里的路径，代码会自动适应并排数量
files = [
    "/home/yangrui/Project/Base-model/local_results/output/Parse/Parse-reshape/GT/144.label.nii.gz",
    "/home/yangrui/Project/Base-model/local_results/output/Parse/Parse-reshape/基础模型-全样本-新200_test/144_pred.nii.gz",
    "/home/yangrui/Project/Base-model/local_results/output/Parse/Parse-reshape/基础模型-少样本5-新200_test/144_pred.nii.gz",
    "/home/yangrui/Project/Base-model/local_results/output/Parse/Parse-reshape/随机初始化-全样本-新200_test/144_pred.nii.gz",
    "/home/yangrui/Project/Base-model/local_results/output/Parse/Parse-reshape/随机初始化-少样本5-新200_test/144_pred.nii.gz"
]

titles = [
    "Ground Truth",
    "Full Sample Base",
    "Few-Shot Base",
    "Full Sample random",
    "Few-Shot random",
]

output_path = "/home/yangrui/Project/Base-model/local_results/output/Parse/vessel_comparison_fine_white_144.png"

# 在这里直接修改 elev (俯仰角) 和 azim (方位角) 来寻找最佳观察角度
# 建议尝试：elev=15, azim=45 或 elev=45, azim=-45 等不同组合来观察微小分支
plot_multiple_3d_fine(files, titles, output_path, elev=0, azim=0)