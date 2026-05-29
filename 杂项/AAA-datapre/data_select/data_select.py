import os
import glob
import shutil
import torch
import numpy as np
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensityRangePercentiles, EnsureType
)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ================= 1. 配置区域 (请修改这里) =================

# 权重路径 (Pre-trained weights)
CKPT_PATH = "/home/yangrui/Project/Base-model/local_results/checkpoints/base.pt"

# 数据集 'all' 的根目录
DATA_ROOT = "/home/yangrui/Project/Base-model/datasets/imageCAS/imageCAS-ROI/all"

# 统一缩放尺寸 (为了特征提取的一致性)
INPUT_SIZE = (128, 128, 128)

# 选取数量配置
NUM_TRAIN_SHOTS = 3  # 训练集选取数量
NUM_VAL_SHOTS = 7  # 验证集选取数量

# 设备
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ================= 2. 模型定义 =================

def get_model():
    model_config = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": [[3, 3, 3]] * 6,
        "strides": [[1, 1, 1]] + [[2, 2, 2]] * 5,
        "upsample_kernel_size": [[2, 2, 2]] * 5,
        "filters": [32, 64, 128, 256, 320, 320],
        "res_block": True
    }
    return DynUNet(**model_config)


def load_weights(model, path):
    print(f"Loading weights from {path}...")
    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return model

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("model.", "").replace("net.", "")
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Success: Weights loaded strictly.")
    except Exception as e:
        print(f"Warning: Strict loading failed ({e}). Trying strict=False...")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Success (Partial). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return model


# ================= 3. 特征提取工具 =================

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.features = torch.mean(output, dim=(2, 3, 4))

        if hasattr(self.model, 'bottleneck'):
            self.hook = self.model.bottleneck.register_forward_hook(hook_fn)
        else:
            raise AttributeError("Model does not have 'bottleneck' layer.")

    def extract(self, x):
        self.features = None
        self.model(x)
        return self.features

    def close(self):
        if self.hook:
            self.hook.remove()


# ================= 4. 数据处理逻辑 =================

def get_transforms():
    return Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1, clip=True),
        Resize(spatial_size=INPUT_SIZE, mode="trilinear"),
        EnsureType(data_type="tensor")
    ])


def get_all_cases(root_dir):
    """仅仅获取目录下所有合法的 Case ID"""
    if not os.path.exists(root_dir):
        raise ValueError(f"Directory not found: {root_dir}")

    cases = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    return cases


def check_existing_test_set(all_dir):
    """
    检查同级是否存在 test 文件夹。
    返回: (exists: bool, excluded_cases: set)
    """
    parent_dir = os.path.dirname(all_dir)
    test_dir = os.path.join(parent_dir, "test")

    if os.path.exists(test_dir):
        test_cases = set([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        return True, test_cases
    return False, set()


def get_image_path(root_dir, case_id):
    """查找病例文件夹下的 img.nii.gz"""
    case_dir = os.path.join(root_dir, case_id)
    img_path = os.path.join(case_dir, f"{case_id}.img.nii.gz")
    if os.path.exists(img_path):
        return img_path

    candidates = glob.glob(os.path.join(case_dir, "*img.nii.gz"))
    if candidates:
        return candidates[0]
    return None


def copy_cases_to_target(case_ids, source_root, target_root):
    """将选中的 Case ID 对应的文件夹复制到目标路径"""
    if not os.path.exists(target_root):
        os.makedirs(target_root)
        print(f"Created directory: {target_root}")

    print(f"Copying {len(case_ids)} cases to -> {target_root}")
    for case_id in tqdm(case_ids, desc=f"To {os.path.basename(target_root)}"):
        src_path = os.path.join(source_root, case_id)
        dst_path = os.path.join(target_root, case_id)

        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)  # 确保覆盖

        shutil.copytree(src_path, dst_path)


# ================= 5. 核心聚类选择算法 =================

def cluster_and_select(features, ids, k, exclude_indices=None):
    if exclude_indices is None:
        exclude_indices = []

    # 1. 过滤数据
    total_samples = len(features)
    mask = np.ones(total_samples, dtype=bool)
    mask[exclude_indices] = False

    subset_features = features[mask]
    subset_ids = [ids[i] for i in range(total_samples) if mask[i]]
    global_indices_map = [i for i in range(total_samples) if mask[i]]

    available_count = len(subset_features)
    if available_count == 0:
        return [], []

    actual_k = min(k, available_count)

    # 2. 执行 K-Means
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(subset_features)
    centers = kmeans.cluster_centers_

    dists = cdist(subset_features, centers, metric='euclidean')

    selected_ids = []
    selected_global_indices = []

    # 3. 选择样本
    for i in range(actual_k):
        in_cluster_indices = np.where(cluster_labels == i)[0]
        if len(in_cluster_indices) > 0:
            min_dist_idx = np.argmin(dists[in_cluster_indices, i])
            subset_idx = in_cluster_indices[min_dist_idx]
            selected_ids.append(subset_ids[subset_idx])
            selected_global_indices.append(global_indices_map[subset_idx])

    return selected_ids, selected_global_indices


# ================= 6. 主程序 =================
import random
def main():
    # =============== 新增：强制确定性锁定 ===============
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 牺牲一点点速度，换取绝对的一致性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # ==================================================

    # --- Step 1: 准备模型 ---
    print("\n--- 1. Initializing Model ---")
    model = get_model().to(DEVICE)
    model = load_weights(model, CKPT_PATH)
    model.eval()

    extractor = FeatureExtractor(model)
    transforms = get_transforms()

    # --- Step 2: 准备数据与判断模式 ---
    print("\n--- 2. Scanning Data & Checking Mode ---")

    # 获取 all 文件夹下所有 ID
    all_case_ids = set(get_all_cases(DATA_ROOT))
    print(f"Total cases in 'all': {len(all_case_ids)}")

    # 检查是否存在 test 文件夹
    test_exists, existing_test_cases = check_existing_test_set(DATA_ROOT)

    final_candidates = []
    generate_new_test_set = False  # 标记是否需要生成新的测试集

    if test_exists:
        print(f"[Mode: Exclude Test] Found existing 'test' folder with {len(existing_test_cases)} cases.")
        # 从候选名单中移除已有的测试集
        final_candidates = sorted(list(all_case_ids - existing_test_cases))
        print(f"Candidates for Train/Val: {len(final_candidates)} (Total - Existing Test)")
    else:
        print("[Mode: Generate Test] 'test' folder NOT found.")
        print("Will use ALL data for selection, and remaining data will become the NEW Test set.")
        final_candidates = sorted(list(all_case_ids))
        generate_new_test_set = True

    # 验证图像路径有效性
    valid_img_paths = []
    valid_ids = []
    for cid in final_candidates:
        p = get_image_path(DATA_ROOT, cid)
        if p:
            valid_img_paths.append(p)
            valid_ids.append(cid)

    if len(valid_ids) < (NUM_TRAIN_SHOTS + NUM_VAL_SHOTS):
        print(f"Error: Not enough data! Need {NUM_TRAIN_SHOTS + NUM_VAL_SHOTS}, found {len(valid_ids)}.")
        return

    # --- Step 3: 提取特征 ---
    print(f"\n--- 3. Extracting Features ({len(valid_ids)} cases) ---")
    features_list = []
    with torch.no_grad():
        for path in tqdm(valid_img_paths, desc="Extracting"):
            try:
                img = transforms(path).unsqueeze(0).to(DEVICE)
                feat = extractor.extract(img)
                features_list.append(feat.cpu().numpy().flatten())
            except Exception as e:
                print(f"Error processing {path}: {e}")
                features_list.append(np.zeros(320))

    features_matrix = np.array(features_list)

    # --- Step 4: 聚类选择 Train ---
    print(f"\n--- 4. Selecting TRAIN sets ({NUM_TRAIN_SHOTS} cases) ---")
    train_ids, train_indices = cluster_and_select(
        features_matrix, valid_ids, k=NUM_TRAIN_SHOTS, exclude_indices=[]
    )
    print(f"Selected Train IDs: {train_ids}")

    # --- Step 5: 聚类选择 Val ---
    print(f"\n--- 5. Selecting VAL sets ({NUM_VAL_SHOTS} cases) ---")
    val_ids, val_indices = cluster_and_select(
        features_matrix, valid_ids, k=NUM_VAL_SHOTS, exclude_indices=train_indices
    )
    print(f"Selected Val IDs: {val_ids}")

    # --- Step 6: 确定 Test 集 (如果是生成模式) ---
    test_ids = []
    if generate_new_test_set:
        print(f"\n--- 6. Identifying TEST sets (Remaining cases) ---")
        # 集合运算：候选集 - Train - Val
        selected_set = set(train_ids) | set(val_ids)
        test_ids = [cid for cid in valid_ids if cid not in selected_set]
        print(f"Remaining {len(test_ids)} cases will be used as Test set.")

    # 清理显存
    extractor.close()
    del model
    torch.cuda.empty_cache()

    # --- Step 7: 创建文件夹并复制 ---
    print("\n--- 7. Creating Directories and Copying ---")
    parent_dir = os.path.dirname(DATA_ROOT)

    train_dir = os.path.join(parent_dir, "train")
    val_dir = os.path.join(parent_dir, "val")
    test_dir = os.path.join(parent_dir, "test")

    # 1. Copy Train
    copy_cases_to_target(train_ids, DATA_ROOT, train_dir)

    # 2. Copy Val
    copy_cases_to_target(val_ids, DATA_ROOT, val_dir)

    # 3. Copy Test (仅当需要生成时)
    if generate_new_test_set and len(test_ids) > 0:
        copy_cases_to_target(test_ids, DATA_ROOT, test_dir)
    elif not generate_new_test_set:
        print("Skipping Test set copy (Existing 'test' folder was preserved).")

    print("\n================ SUCCESS ================")
    print(f"Train Dir: {train_dir} ({len(train_ids)} cases)")
    print(f"Val Dir:   {val_dir} ({len(val_ids)} cases)")
    if generate_new_test_set:
        print(f"Test Dir:  {test_dir} ({len(test_ids)} cases)")
    else:
        print(f"Test Dir:  (Existing folder kept)")


if __name__ == "__main__":
    main()