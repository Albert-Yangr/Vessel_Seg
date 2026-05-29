import logging, warnings, os, sys
from pathlib import Path

# 🌟 核心修复 1：把你的项目根目录加入环境变量，直接复用原生的读取和预处理逻辑
PROJECT_ROOT = "/home/yangrui/Project/Base-model"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 暴力限制所有底层 C 库的线程数，防止高配服务器上的 OpenBLAS 崩溃
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import SimpleITK as sitk
import hydra
from omegaconf import DictConfig

from scipy.ndimage import binary_opening
from monai.inferers import sliding_window_inference

# 🌟 核心修复 2：导入你自己的工具包，消灭坐标系错乱
from utils.io import determine_reader_writer
from utils.dataset import generate_transforms

logger = logging.getLogger(__name__)

# ==========================================
# 1. 配置参数区
# ==========================================
IMAGE_FOLDER = "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-refile/test/91"
path_parts = Path(IMAGE_FOLDER).parts
dataset = path_parts[-4]

OUTPUT_FOLDER = "/home/yangrui/Project/Base-model/local_results/feature_analysis/" + dataset + "_FullVolume"

CKPT_PATHS = {
    "Anchor_5": "/home/yangrui/Project/Base-model/local_results/checkpoints/CAS2023/CAS2023-refile/自适应2_基础500/Epoch36-0.7952.ckpt",
    "Slice_5": "/home/yangrui/Project/Base-model/local_results/checkpoints/CAS2023/CAS2023-refile/双分支-切片基线/Epoch42-0.7879.ckpt"
}

NUM_SAMPLES = 5000
PATCH_SIZE = 128
VESSEL_SIZE_THRESHOLD = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. 核心功能函数
# ==========================================
def pad_tensor_to_multiple(tensor, multiple=128):
    """将 Tensor 的最后三个维度补齐到 multiple 的整数倍"""
    _, _, D, H, W = tensor.shape
    pad_d = (multiple - D % multiple) % multiple
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple

    pad = (0, pad_w, 0, pad_h, 0, pad_d)
    padded = F.pad(tensor, pad, mode='constant', value=tensor.min().item())
    return padded, (D, H, W)


def pad_mask_to_multiple(mask, multiple=128):
    """将 Mask Numpy 数组补齐到 multiple 的整数倍"""
    D, H, W = mask.shape
    pad_d = (multiple - D % multiple) % multiple
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple

    pad = (0, pad_w, 0, pad_h, 0, pad_d)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    padded = F.pad(mask_t, pad, mode='constant', value=0).squeeze().numpy().astype(np.uint8)
    return padded


def load_model(cfg, ckpt_path):
    model = hydra.utils.instantiate(cfg.model).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    new_state = {}
    for k, v in state.items():
        if "projection_head" in k or "aux_upsamples" in k or "aux_output_block" in k: continue
        new_k = k.replace('model.', '').replace('net.', '').replace('module.', '').replace('base_model.', '').replace(
            'base_', '')
        new_state[new_k] = v
    try:
        model.load_state_dict(new_state, strict=True)
    except RuntimeError:
        model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


# ==========================================
# 3. 绘图辅助函数
# ==========================================
def plot_and_save_histogram(combined_features, labels_array, owner_array, model_names, save_path, title_prefix=""):
    fig_hist, axes_hist = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), sharex=True, sharey=True)
    if len(model_names) == 1: axes_hist = [axes_hist]
    for ax, name in zip(axes_hist, model_names):
        mask = (owner_array == name)
        feats = combined_features[mask]
        lbls = labels_array[mask]
        true_vessel_mask = (lbls == 1) | (lbls == 3)
        if np.any(true_vessel_mask):
            vessel_centroid = np.mean(feats[true_vessel_mask], axis=0, keepdims=True)
            sims = cosine_similarity(feats, vessel_centroid).flatten()
            sim_bg = sims[(lbls == 0) | (lbls == 2)]
            sim_fg = sims[(lbls == 1) | (lbls == 3)]
            ax.hist(sim_bg, bins=40, alpha=0.5, color='#4C72B0', label='Background', density=True)
            ax.hist(sim_fg, bins=40, alpha=0.7, color='#C44E52', label='True Vessel', density=True)
        ax.set_title(f"[{name}]\n{title_prefix} Cosine Sim. to Vessel", fontsize=14)
        ax.set_xlabel("Cosine Similarity (-1 to 1)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig_hist.savefig(save_path, dpi=300)
    plt.close(fig_hist)


def plot_and_save_tsne(reduced_features, labels_array, owner_array, model_names, save_path, title_prefix=""):
    fig_bin, axes_bin = plt.subplots(1, len(model_names), figsize=(7 * len(model_names), 7), sharex=True, sharey=True)
    if len(model_names) == 1: axes_bin = [axes_bin]
    for ax, name in zip(axes_bin, model_names):
        mask = (owner_array == name)
        pts = reduced_features[mask]
        lbls = labels_array[mask]
        bg_pts = pts[(lbls == 0) | (lbls == 2)]
        vessel_pts = pts[(lbls == 1) | (lbls == 3)]
        ax.scatter(bg_pts[:, 0], bg_pts[:, 1], c='#4C72B0', label='Background', s=15, alpha=0.3, edgecolors='none')
        ax.scatter(vessel_pts[:, 0], vessel_pts[:, 1], c='#C44E52', label='True Vessel', s=25, alpha=0.7,
                   edgecolors='white', linewidths=0.5)
        ax.set_title(f"[{name}]\n{title_prefix} Feature Space", fontsize=16)
        ax.legend(loc='best', fontsize=12, markerscale=2)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks([]);
        ax.set_yticks([])
    plt.tight_layout()
    fig_bin.savefig(save_path, dpi=300)
    plt.close(fig_bin)


def plot_and_save_tsne_4class(reduced_features, labels_array, owner_array, model_names, save_path, title_prefix=""):
    fig_4c, axes_4c = plt.subplots(1, len(model_names), figsize=(7 * len(model_names), 7), sharex=True, sharey=True)
    if len(model_names) == 1: axes_4c = [axes_4c]
    color_map = {0: '#E0E0E0', 1: '#2CA02C', 2: '#D62728', 3: '#FF7F0E'}
    label_map = {0: 'TN (True BG)', 1: 'TP (True Vessel)', 2: 'FP (Noise)', 3: 'FN (Missed)'}
    zorder_map = {0: 1, 1: 3, 2: 4, 3: 4}
    alpha_map = {0: 0.3, 1: 0.7, 2: 0.9, 3: 0.9}
    for ax, name in zip(axes_4c, model_names):
        mask = (owner_array == name)
        pts = reduced_features[mask]
        lbls = labels_array[mask]
        for class_id in [0, 1, 3, 2]:
            c_mask = (lbls == class_id)
            if not np.any(c_mask): continue
            ax.scatter(pts[c_mask, 0], pts[c_mask, 1], c=color_map[class_id], label=label_map[class_id],
                       s=15 if class_id == 0 else 25, alpha=alpha_map[class_id], zorder=zorder_map[class_id],
                       edgecolors='white' if class_id in [2, 3] else 'none', linewidths=0.5)
        ax.set_title(f"[{name}]\n{title_prefix} 4-Class Diagnosis", fontsize=16)
        ax.legend(loc='best', fontsize=12, markerscale=2)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks([]);
        ax.set_yticks([])
    plt.tight_layout()
    fig_4c.savefig(save_path, dpi=300)
    plt.close(fig_4c)


def plot_and_save_tsne_3class_size(reduced_features, labels_array, owner_array, model_names, save_path,
                                   title_prefix=""):
    fig_3c, axes_3c = plt.subplots(1, len(model_names), figsize=(7 * len(model_names), 7), sharex=True, sharey=True)
    if len(model_names) == 1: axes_3c = [axes_3c]
    color_map = {0: '#E0E0E0', 1: '#3498DB', 2: '#A93226'}
    label_map = {0: 'Background', 1: 'Small Vessel (Thin)', 2: 'Large Vessel (Thick)'}
    zorder_map = {0: 1, 1: 3, 2: 4}
    alpha_map = {0: 0.3, 1: 0.8, 2: 0.8}
    for ax, name in zip(axes_3c, model_names):
        mask = (owner_array == name)
        pts = reduced_features[mask]
        lbls = labels_array[mask]
        for class_id in [0, 1, 2]:
            c_mask = (lbls == class_id)
            if not np.any(c_mask): continue
            ax.scatter(pts[c_mask, 0], pts[c_mask, 1], c=color_map[class_id], label=label_map[class_id],
                       s=15 if class_id == 0 else 30, alpha=alpha_map[class_id], zorder=zorder_map[class_id],
                       edgecolors='white' if class_id > 0 else 'none', linewidths=0.5)
        ax.set_title(f"[{name}]\n{title_prefix} Vessel Size Distribution", fontsize=16)
        ax.legend(loc='upper right', fontsize=12, markerscale=2)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks([]);
        ax.set_yticks([])
    plt.tight_layout()
    fig_3c.savefig(save_path, dpi=300)
    plt.close(fig_3c)


# ==========================================
# 4. 主程序
# ==========================================
@hydra.main(config_path="../../configs", config_name="inference/tem_infer", version_base="1.3.2")
def run_analysis(cfg: DictConfig):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 🌟 新增：专门存放标签的独立文件夹
    label_save_dir = Path(OUTPUT_FOLDER) / "label"
    label_save_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list(Path(IMAGE_FOLDER).rglob("*.img.nii.gz"))
    if not img_paths: return print("❌ 未找到图像")

    img_path = img_paths[0]
    mask_path = Path(str(img_path).replace(".img.nii.gz", ".label.nii.gz"))

    # 读取原始数据
    io_cls = determine_reader_writer('nii.gz')
    rw = io_cls()
    img_np_raw = rw.read_images(str(img_path))[0].astype(np.float32)
    mask_np_raw = rw.read_images(str(mask_path))[0].astype(np.uint8)

    # 🌟 辅助函数：携带原生空间信息保存 NIfTI
    itk_ref_image = sitk.ReadImage(str(mask_path))

    def save_label_nii(np_array, file_name):
        img = sitk.GetImageFromArray(np_array.astype(np.uint8))
        img.CopyInformation(itk_ref_image)  # 完全复制空间源信息
        sitk.WriteImage(img, str(label_save_dir / file_name))

    print(f"📁 正在保存 GT 相关标签到 {label_save_dir}...")
    save_label_nii(mask_np_raw, "00_GT_Full.nii.gz")

    # 预先在原生尺寸下计算出真实的粗细血管并保存
    z_m, y_m, x_m = np.ogrid[-VESSEL_SIZE_THRESHOLD:VESSEL_SIZE_THRESHOLD + 1,
                    -VESSEL_SIZE_THRESHOLD:VESSEL_SIZE_THRESHOLD + 1, -VESSEL_SIZE_THRESHOLD:VESSEL_SIZE_THRESHOLD + 1]
    spherical_structure = (x_m ** 2 + y_m ** 2 + z_m ** 2) <= (VESSEL_SIZE_THRESHOLD ** 2)

    thick_gt_raw = binary_opening(mask_np_raw > 0, structure=spherical_structure)
    thin_gt_raw = (mask_np_raw > 0) ^ thick_gt_raw

    save_label_nii(thick_gt_raw, "00_GT_Thick.nii.gz")
    save_label_nii(thin_gt_raw, "00_GT_Thin.nii.gz")

    # 执行预处理 Transform
    trans = generate_transforms(cfg.transforms_config)
    data_in = trans(img_np_raw)
    x_tensor = torch.from_numpy(data_in) if isinstance(data_in, np.ndarray) else data_in
    if x_tensor.ndim == 3:
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)
    elif x_tensor.ndim == 4:
        x_tensor = x_tensor.unsqueeze(0)
    x_tensor = x_tensor.float()

    all_model_data = {}

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for name, ckpt in CKPT_PATHS.items():
                print(f"\n{'=' * 50}")
                print(f"⚙️ 正在评估模型: [{name}]")
                model = load_model(cfg, ckpt)

                # ==============================================================
                # 1. 官方推理路径
                # ==============================================================
                print("🔍 [步骤1] 执行官方重叠推理...")
                val_outputs = sliding_window_inference(
                    inputs=x_tensor.to(DEVICE),
                    roi_size=(128, 128, 128),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5
                )
                val_preds = (torch.sigmoid(val_outputs) > 0.5).cpu().squeeze().numpy().astype(np.uint8)

                intersection = np.sum(val_preds * mask_np_raw)
                dice_score = (2.0 * intersection) / (np.sum(val_preds) + np.sum(mask_np_raw) + 1e-5)
                print(f" 🏆 [0.5 重叠混合] 官方推理 Dice 得分: {dice_score * 100:.2f}%")

                # 🌟 保存各个模型的预测结果到 label 文件夹
                save_label_nii(val_preds, f"01_Pred_{name}.nii.gz")

                # ==============================================================
                # 2. 特征截获路径
                # ==============================================================
                print("⚙️ [步骤2] 执行纯净特征挂载与截获...")
                x_padded, orig_shape = pad_tensor_to_multiple(x_tensor, PATCH_SIZE)

                # 同步 Padding 所有的标签矩阵供特征提取使用
                mask_padded = pad_mask_to_multiple(mask_np_raw, PATCH_SIZE)
                preds_padded = pad_mask_to_multiple(val_preds, PATCH_SIZE)

                _, _, Z, Y, X = x_padded.shape

                full_features = {}
                is_first_patch = True

                extracted_features = {}

                def get_hook(layer_name):
                    def hook(module, input, output):
                        extracted_features[layer_name] = output.detach().cpu().numpy()[0]

                    return hook

                h32 = model.upsamples[2].register_forward_hook(get_hook('Layer_032'))
                h64 = model.upsamples[3].register_forward_hook(get_hook('Layer_064'))
                h128 = model.upsamples[4].register_forward_hook(get_hook('Layer_128'))

                for z in range(0, Z, PATCH_SIZE):
                    for y in range(0, Y, PATCH_SIZE):
                        for x in range(0, X, PATCH_SIZE):
                            patch = x_padded[:, :, z:z + PATCH_SIZE, y:y + PATCH_SIZE, x:x + PATCH_SIZE].to(DEVICE)
                            _ = model(patch)

                            if is_first_patch:
                                for ln in ['Layer_032', 'Layer_064', 'Layer_128']:
                                    c, pz, py, px = extracted_features[ln].shape
                                    scale = PATCH_SIZE // pz
                                    full_features[ln] = np.zeros((c, Z // scale, Y // scale, X // scale),
                                                                 dtype=np.float16)
                                is_first_patch = False

                            for ln in ['Layer_032', 'Layer_064', 'Layer_128']:
                                feat = extracted_features[ln]
                                scale = PATCH_SIZE // feat.shape[1]
                                fz, fy, fx = z // scale, y // scale, x // scale
                                fsz = PATCH_SIZE // scale
                                full_features[ln][:, fz:fz + fsz, fy:fy + fsz, fx:fx + fsz] = feat.astype(np.float16)

                            extracted_features.clear()

                h32.remove();
                h64.remove();
                h128.remove()

                all_model_data[name] = {
                    'preds': preds_padded,
                    'Layer_032': full_features['Layer_032'],
                    'Layer_064': full_features['Layer_064'],
                    'Layer_128': full_features['Layer_128']
                }

    # =========================================================================
    # 下游 t-SNE 分析管线
    # =========================================================================
    patch_mask = mask_padded
    layers_config = [("Layer_032", 4), ("Layer_064", 2), ("Layer_128", 1)]
    patch_mask_tensor = torch.from_numpy(patch_mask).float().unsqueeze(0).unsqueeze(0)

    for layer_name, scale_factor in layers_config:
        print(f"\n🚀 开始解析全图多尺度特征 -> 【{layer_name}】 (缩放系数: 1/{scale_factor})")

        layer_out_dir = Path(OUTPUT_FOLDER) / layer_name
        layer_out_dir.mkdir(parents=True, exist_ok=True)

        if scale_factor > 1:
            down_mask = F.max_pool3d(patch_mask_tensor, kernel_size=scale_factor, stride=scale_factor)
            down_mask = down_mask.squeeze().numpy().astype(np.uint8)
        else:
            down_mask = patch_mask

        vessel_coords = np.argwhere(down_mask == 1)
        bg_coords = np.argwhere(down_mask == 0)

        np.random.seed(42)
        idx_v = np.random.choice(len(vessel_coords), min(NUM_SAMPLES, len(vessel_coords)), replace=False)
        idx_b = np.random.choice(len(bg_coords), min(NUM_SAMPLES, len(bg_coords)), replace=False)
        sampled_v_coords = vessel_coords[idx_v]
        sampled_b_coords = bg_coords[idx_b]

        combined_features, labels_list, owner_list = [], [], []

        for name in CKPT_PATHS.keys():
            feat_map = all_model_data[name][layer_name]
            feat_v = feat_map[:, sampled_v_coords[:, 0], sampled_v_coords[:, 1], sampled_v_coords[:, 2]].T
            combined_features.append(feat_v)
            labels_list.extend([1] * len(feat_v))
            owner_list.extend([name] * len(feat_v))

            feat_b = feat_map[:, sampled_b_coords[:, 0], sampled_b_coords[:, 1], sampled_b_coords[:, 2]].T
            combined_features.append(feat_b)
            labels_list.extend([0] * len(feat_b))
            owner_list.extend([name] * len(feat_b))

        X = np.vstack(combined_features)
        y = np.array(labels_list)
        owners = np.array(owner_list)
        model_names = list(CKPT_PATHS.keys())

        hist_path = layer_out_dir / f"02_{layer_name}_Cosine_Sim.png"
        plot_and_save_histogram(X, y, owners, model_names, hist_path, title_prefix=f"{layer_name}")

        print(f" 🌌 正在运行 t-SNE 降维 (采样点数: {len(X)})...", end=" ", flush=True)
        X_reduced = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', learning_rate='auto',
                         n_jobs=4).fit_transform(X)
        plot_and_save_tsne(X_reduced, y, owners, model_names, layer_out_dir / f"01_{layer_name}_tSNE_2class.png",
                           title_prefix=f"{layer_name}")
        print("✅ 完成。")

    # =========================================================================
    # 🌟 定制 1：Layer_128 四分类 (TP/TN/FP/FN)
    # =========================================================================
    print(f"\n🔬 启动全图 Layer_128 四分类查错...")
    layer_128_out_dir = Path(OUTPUT_FOLDER) / "Layer_128"
    all_features_4c, labels_list_4c, owner_list_4c = [], [], []

    np.random.seed(42)
    for name in CKPT_PATHS.keys():
        preds = all_model_data[name]['preds']
        feats_np = all_model_data[name]['Layer_128']

        tp_coords = np.argwhere((preds == 1) & (patch_mask == 1))
        tn_coords = np.argwhere((preds == 0) & (patch_mask == 0))
        fp_coords = np.argwhere((preds == 1) & (patch_mask == 0))
        fn_coords = np.argwhere((preds == 0) & (patch_mask == 1))

        def sample_feats_4c(coords, label_id):
            if len(coords) == 0: return
            idx = np.random.choice(len(coords), min(NUM_SAMPLES, len(coords)), replace=False)
            sampled_c = coords[idx]
            sampled_f = feats_np[:, sampled_c[:, 0], sampled_c[:, 1], sampled_c[:, 2]].T
            all_features_4c.append(sampled_f)
            labels_list_4c.extend([label_id] * len(sampled_f))
            owner_list_4c.extend([name] * len(sampled_f))

        sample_feats_4c(tn_coords, 0);
        sample_feats_4c(tp_coords, 1)
        sample_feats_4c(fp_coords, 2);
        sample_feats_4c(fn_coords, 3)

    combined_features_4c = np.vstack(all_features_4c)
    print(f" 🌌 运行 4 分类 t-SNE 降维运算 (点数: {len(combined_features_4c)})...", end=" ", flush=True)
    X_reduced_4c = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', learning_rate='auto',
                        n_jobs=4).fit_transform(combined_features_4c)
    plot_and_save_tsne_4class(X_reduced_4c, np.array(labels_list_4c), np.array(owner_list_4c), list(CKPT_PATHS.keys()),
                              layer_128_out_dir / "03_Layer_128_tSNE_4class_diagnosis.png", title_prefix="Layer_128")
    print("✅ 完成。")

    # =========================================================================
    # 🌟 定制 2：Layer_128 全图形态学粗细分类特征
    # =========================================================================
    print(f"\n📏 提取全图粗细血管特征...")

    # 将前面生成的真实粗细血管 Padding，以对齐特征空间
    thick_mask_np = pad_mask_to_multiple(thick_gt_raw, PATCH_SIZE)
    thin_mask_np = pad_mask_to_multiple(thin_gt_raw, PATCH_SIZE)

    all_features_3c, labels_list_3c, owner_list_3c = [], [], []

    np.random.seed(42)
    for name in CKPT_PATHS.keys():
        feats_np = all_model_data[name]['Layer_128']
        bg_coords = np.argwhere(patch_mask == 0)
        thin_coords = np.argwhere(thin_mask_np == True)
        thick_coords = np.argwhere(thick_mask_np == True)

        def sample_feats_3c(coords, label_id):
            if len(coords) == 0: return
            idx = np.random.choice(len(coords), min(NUM_SAMPLES, len(coords)), replace=False)
            sampled_c = coords[idx]
            sampled_f = feats_np[:, sampled_c[:, 0], sampled_c[:, 1], sampled_c[:, 2]].T
            all_features_3c.append(sampled_f)
            labels_list_3c.extend([label_id] * len(sampled_f))
            owner_list_3c.extend([name] * len(sampled_f))

        sample_feats_3c(bg_coords, 0);
        sample_feats_3c(thin_coords, 1);
        sample_feats_3c(thick_coords, 2)

    combined_features_3c = np.vstack(all_features_3c)
    print(f" 🌌 运行粗细三分类 t-SNE 降维运算 (点数: {len(combined_features_3c)})...", end=" ", flush=True)
    X_reduced_3c = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', learning_rate='auto',
                        n_jobs=4).fit_transform(combined_features_3c)
    plot_and_save_tsne_3class_size(X_reduced_3c, np.array(labels_list_3c), np.array(owner_list_3c),
                                   list(CKPT_PATHS.keys()),
                                   layer_128_out_dir / "04_Layer_128_tSNE_3class_size_diagnosis.png",
                                   title_prefix="Layer_128")
    print("✅ 完成。")

    print(f"\n🎉 全图分析圆满结束！所有标签均已保存至: {label_save_dir}")


if __name__ == "__main__":
    run_analysis()