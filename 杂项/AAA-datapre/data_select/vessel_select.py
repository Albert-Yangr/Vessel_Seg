import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import nibabel as nib
import shutil
import math
import random

from sympy import false
from tqdm import tqdm
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# ================= 1. 深度科研配置 (Configuration) =================
CONFIG = {
    # 路径配置
    "ckpt_path": "/home/yangrui/Project/Base-model/local_results/checkpoints/base.pt",
    "data_root": "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-S/all",
    "test_dir": "/home/yangrui/Project/Base-model/datasets/CAS2023/CAS2023-S/test",

    # 采样参数
    "num_train": 5,  # 第一轮聚类：选 3 个全局代表
    "num_val": 0,  # 第二轮聚类：在剩下的数据里选 7 个局部代表
    "include_test": False,

    # 极端个例排除参数
    "outlier_fraction": 0.05,  # 排除距离分布中心最远的 5% 的样本 (视为极端/异常数据)

    # 模型与计算参数
    "roi_size": (128, 128, 128),
    "gpu_ids": [0],  # 并行GPU列表 (即使数量改变，结果也不会变)

    # STARS 算法参数
    "fingerprint_dim": (8, 8, 8),
    "style_weight": 1.0
}


# ================= 2. 多进程特征提取 Worker (含确定性修复) =================

def _worker_process(gpu_id, case_list, config, return_list):
    """
    STARS Feature Extractor
    已加入确定性控制和精度截断，确保跨设备一致性。
    """
    # === 强制 PyTorch 确定性模式 ===
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 锁定所有随机种子
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 禁用 benchmark，启用确定性算法
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")

    print(f"[Worker-{gpu_id}] 启动 STARS 特征提取 (Deterministic)...")

    try:
        model = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=1,
            kernel_size=[[3, 3, 3]] * 6, strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
            upsample_kernel_size=[[2, 2, 2]] * 5, filters=[32, 64, 128, 256, 320, 320], res_block=True
        )
        ckpt = torch.load(config["ckpt_path"], map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()}, strict=False)
        model.to(device).eval()
    except Exception as e:
        print(f"[Worker-{gpu_id}] 模型加载失败: {e}")
        return

    def predictor(x):
        return torch.sigmoid(model(x))

    local_results = []

    for cid in tqdm(case_list, desc=f"GPU-{gpu_id}", position=int(gpu_id)):
        img_p = os.path.join(config["data_root"], cid, f"{cid}.img.nii.gz")
        if not os.path.exists(img_p): continue

        try:
            img = nib.load(img_p).get_fdata()
            # 基础预处理
            img = (img - img.mean()) / (img.std() + 1e-8)
            img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                prob_map = sliding_window_inference(img_t, config["roi_size"], 4, predictor, device=device)

                # 1. 结构特征
                topo_feat = F.adaptive_avg_pool3d(prob_map, config["fingerprint_dim"])
                topo_vec = topo_feat.cpu().numpy().flatten()

                # 2. 风格特征
                weights = prob_map
                sum_w = weights.sum() + 1e-8
                w_mean = (img_t * weights).sum() / sum_w
                w_std = torch.sqrt(((img_t - w_mean) ** 2 * weights).sum() / sum_w)
                style_vec = np.array([w_mean.item(), w_std.item()])

            # 简单的质量过滤
            if prob_map.mean().item() < 0.0005:
                continue

            # === 物理精度截断 (关键步骤) ===
            # 切除小数点后6位以外的波动，抵抗 GPU 并行求和带来的微小差异
            topo_vec = np.round(topo_vec, decimals=6)
            style_vec = np.round(style_vec, decimals=6)

            local_results.append({
                'id': cid,
                'topo_feat': topo_vec,
                'style_feat': style_vec,
                'src': os.path.join(config["data_root"], cid)
            })

        except Exception as e:
            continue

    return_list.extend(local_results)


# ================= 3. 核心分发器 (STARS Logic: Double Clustering) =================

class StarsSelector:
    def __init__(self):
        self.dataset_base = os.path.dirname(CONFIG["data_root"])
        self.output_dir = self.dataset_base
        self.registry = []  # 正常的候选样本
        self.outliers = []  # 剔除的极端样本
        self.has_test = os.path.exists(CONFIG["test_dir"]) and len(os.listdir(CONFIG["test_dir"])) > 0

    def run_parallel_extraction(self):
        all_cases = sorted([d for d in os.listdir(CONFIG["data_root"])
                            if os.path.isdir(os.path.join(CONFIG["data_root"], d))])

        existing_test = set(os.listdir(CONFIG["test_dir"])) if self.has_test else set()
        candidate_cases = [c for c in all_cases if c not in existing_test]

        print(f"Total Candidates: {len(candidate_cases)} | Excluded Existing Test: {len(existing_test)}")

        gpu_ids = CONFIG["gpu_ids"]
        chunk_size = math.ceil(len(candidate_cases) / len(gpu_ids))
        chunks = [candidate_cases[i:i + chunk_size] for i in range(0, len(candidate_cases), chunk_size)]

        manager = mp.Manager()
        return_list = manager.list()
        processes = []

        mp.set_start_method('spawn', force=True)
        print(">>> Starting STARS Feature Extraction...")

        for rank, gpu_id in enumerate(gpu_ids):
            if rank >= len(chunks): break
            p = mp.Process(target=_worker_process, args=(gpu_id, chunks[rank], CONFIG, return_list))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # === 强制排序 ===
        # 确保无论多进程怎么乱序，最终进入算法的列表顺序永远一致
        self.registry = sorted(list(return_list), key=lambda x: x['id'])
        print(f"\n[Extraction Done] Valid Samples: {len(self.registry)}")

    def execute_selection(self):
        if not self.registry: return

        # =========================================================
        # 第一阶段：数据清洗与标准化 (For Train Selection)
        # =========================================================

        topo_X = np.array([d['topo_feat'] for d in self.registry])
        style_X = np.array([d['style_feat'] for d in self.registry])

        # 全局标准化
        scaler_topo = StandardScaler()
        scaler_style = StandardScaler()

        topo_norm = scaler_topo.fit_transform(topo_X)
        style_norm = scaler_style.fit_transform(style_X)

        # 特征融合
        combined_X = np.concatenate([topo_norm, style_norm * CONFIG["style_weight"]], axis=1)

        # =========================================================
        # 极端个例排除 (Outlier Rejection)
        # =========================================================
        print(f"\n[STARS Filter] Screening for outliers (Threshold: Top {CONFIG['outlier_fraction'] * 100}%)...")

        # 计算全局中心
        global_center = np.mean(combined_X, axis=0).reshape(1, -1)
        dists_to_global = cdist(combined_X, global_center, metric='euclidean').flatten()

        # 确定阈值 (例如 95分位数)
        dist_threshold = np.percentile(dists_to_global, (1 - CONFIG["outlier_fraction"]) * 100)

        normal_indices = np.where(dists_to_global <= dist_threshold)[0]
        outlier_indices = np.where(dists_to_global > dist_threshold)[0]

        # 分离数据
        full_registry = self.registry
        self.registry = [full_registry[i] for i in normal_indices]  # 更新 self.registry 为纯净版
        self.outliers = [full_registry[i] for i in outlier_indices]  # 暂存异常值
        combined_X_clean = combined_X[normal_indices]  # 更新特征矩阵

        print(f"  > Clean Candidates: {len(self.registry)} | Outliers: {len(self.outliers)}")

        # =========================================================
        # 第二阶段：挑选 Train (Primary Clustering)
        # =========================================================
        print(f"\n[Selection Round 1] Selecting {CONFIG['num_train']} TRAIN prototypes...")

        # 在纯净数据上聚类
        kmeans_train = KMeans(n_clusters=CONFIG['num_train'], random_state=42, n_init=20).fit(combined_X_clean)
        centers_train = kmeans_train.cluster_centers_

        train_list = []
        train_ids = set()

        for i in range(CONFIG['num_train']):
            cluster_indices = np.where(kmeans_train.labels_ == i)[0]
            if len(cluster_indices) == 0: continue

            cluster_feats = combined_X_clean[cluster_indices]
            # 找离当前 Cluster 中心最近的样本
            dists = cdist(cluster_feats, [centers_train[i]], metric='euclidean').flatten()

            best_local_idx = np.argmin(dists)
            best_global_idx = cluster_indices[best_local_idx]  # Index relative to clean registry

            item = self.registry[best_global_idx]
            item['cluster_id'] = f"Train-{i}"
            item['dist'] = dists[best_local_idx]

            train_list.append(item)
            train_ids.add(item['id'])

        # =========================================================
        # 第三阶段：挑选 Val (Secondary Representative Selection)
        # =========================================================
        # 关键逻辑：在排除 Train 后的剩余池中，重新进行特征归一化和聚类，寻找最具代表性的样本

        remainders = [d for d in self.registry if d['id'] not in train_ids]
        val_list = []
        val_ids = set()

        if CONFIG["num_val"] > 0 and len(remainders) > 0:
            print(
                f"\n[Selection Round 2] Selecting {CONFIG['num_val']} VAL representatives from {len(remainders)} remainders...")

            # --- 重新构建剩余数据的特征矩阵 (Local Context) ---
            r_topo = np.array([d['topo_feat'] for d in remainders])
            r_style = np.array([d['style_feat'] for d in remainders])

            # 重新标准化 (让分布适应当前的剩余数据池，拉开差异)
            r_topo_norm = StandardScaler().fit_transform(r_topo)
            r_style_norm = StandardScaler().fit_transform(r_style)
            r_combined = np.concatenate([r_topo_norm, r_style_norm * CONFIG["style_weight"]], axis=1)

            # --- 二次聚类 (K-Means on Remainders) ---
            actual_k = min(len(remainders), CONFIG["num_val"])

            # 使用不同的 random_state 避免偶发巧合，保证独立性
            kmeans_val = KMeans(n_clusters=actual_k, random_state=2023, n_init=20).fit(r_combined)
            centers_val = kmeans_val.cluster_centers_

            for i in range(actual_k):
                cluster_indices = np.where(kmeans_val.labels_ == i)[0]
                if len(cluster_indices) == 0: continue

                cluster_feats = r_combined[cluster_indices]
                # 找离当前 Val-Cluster 中心最近的
                dists = cdist(cluster_feats, [centers_val[i]], metric='euclidean').flatten()

                best_local_idx = np.argmin(dists)

                item = remainders[cluster_indices[best_local_idx]]
                item['cluster_id'] = f"Val-{i}"
                item['dist'] = dists[best_local_idx]

                val_list.append(item)
                val_ids.add(item['id'])

        # =========================================================
        # 第四阶段：处理 Test (The Rest + Outliers)
        # =========================================================
        test_list = []
        if CONFIG["include_test"] or not self.has_test:
            # Test = Clean中未被Train/Val选中的 + 所有的 Outliers
            clean_test = [d for d in remainders if d['id'] not in val_ids]

            # 极端样本必须放入 Test，用以测试模型的鲁棒性
            final_test_candidates = clean_test + self.outliers

            target_test_dir = CONFIG["test_dir"]
            if not os.path.exists(target_test_dir): os.makedirs(target_test_dir)

            print(f"Populating Test set ({len(final_test_candidates)} cases)...")
            print(f"  (Clean Remainder: {len(clean_test)} + Outliers: {len(self.outliers)})")

            for item in tqdm(final_test_candidates, desc="Copying Test"):
                dst = os.path.join(target_test_dir, item['id'])
                if not os.path.exists(dst):
                    shutil.copytree(item['src'], dst)
            test_list = final_test_candidates
        else:
            print("Existing Test set preserved.")

        # 物理分发
        self._distribute_files(train_list, "train-select")
        self._distribute_files(val_list, "val")

        return train_list, val_list, test_list

    def _distribute_files(self, data_list, folder_name):
        path = os.path.join(self.dataset_base, folder_name)
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path)

        print(f"Distributing {folder_name} ({len(data_list)} cases)...")
        for item in data_list:
            shutil.copytree(item['src'], os.path.join(path, item['id']))


# ================= 4. 主程序入口 =================

if __name__ == "__main__":
    selector = StarsSelector()

    # 1. 提取特征
    selector.run_parallel_extraction()

    # 2. 聚类并分发
    train_data, val_data, test_data = selector.execute_selection()

    # 3. 生成详细报告
    dim_str = "-".join(map(str, CONFIG["fingerprint_dim"]))
    filename = f"STARS_Report_dim{dim_str}_train{CONFIG['num_train']}_val{CONFIG['num_val']}.txt"
    report_path = os.path.join(selector.output_dir, filename)

    with open(report_path, "w") as f:
        f.write("=== STARS (Structure-Texture Aware Representative Sampling) Report ===\n")
        f.write(f"Timestamp: {os.times()}\n")
        f.write(f"Config: Dim={CONFIG['fingerprint_dim']}, Train={CONFIG['num_train']}, Val={CONFIG['num_val']}\n")
        f.write(f"Style Weight: {CONFIG['style_weight']}\n")
        f.write(f"Outlier Fraction: {CONFIG['outlier_fraction']} (Excluded from Train/Val candidates)\n\n")

        f.write(f"--- DETECTED OUTLIERS ({len(selector.outliers)}) ---\n")
        f.write("These cases were statistically too far from the center (Top 5%) and excluded from training:\n")
        f.write(", ".join([d['id'] for d in selector.outliers]))
        f.write("\n\n")

        f.write(f"--- SELECTED TRAIN PROTOTYPES (k={CONFIG['num_train']}) ---\n")
        for item in train_data:
            f.write(f"ID: {item['id']}\n")
            f.write(f"  Cluster: {item['cluster_id']}\n")
            f.write(f"  Style(Mean/Std): {item['style_feat'][0]:.2f} / {item['style_feat'][1]:.2f}\n")
            f.write(f"  Dist to Center: {item['dist']:.4f}\n\n")

        f.write(f"--- SELECTED VAL REPRESENTATIVES (k={CONFIG['num_val']}) ---\n")
        f.write("Selected via secondary clustering on the remaining data:\n")
        for item in val_data:
            f.write(f"ID: {item['id']}\n")
            f.write(f"  Cluster: {item['cluster_id']}\n")
            f.write(f"  Style(Mean/Std): {item['style_feat'][0]:.2f} / {item['style_feat'][1]:.2f}\n")
            f.write(f"  Dist to Local Center: {item['dist']:.4f}\n")

    print(f"\n[Done] 筛选完成！报告已生成: {report_path}")