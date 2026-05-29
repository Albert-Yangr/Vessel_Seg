"""
模型推理脚本 - 用于3D血管分割的零样本推理
支持.pt和.ckpt权重文件
已优化：
1. 增强的文件名匹配逻辑 (支持 .img.nii.gz <-> .label.nii.gz 等复杂匹配)
2. 修复 SimpleITK 保存时的后缀错误 (区分查找key和文件extension)
"""
import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

from utils.new.data import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_model(cfg, device):
    """
    加载模型权重，智能处理.pt和.ckpt格式文件
    """
    ckpt_path = Path(cfg.ckpt_path)
    logger.info(f"Loading models from {ckpt_path}.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"加载模型文件失败: {e}")
        raise

    model = hydra.utils.instantiate(cfg.model)
    state_dict_to_load = None

    if 'state_dict' in ckpt:
        logger.info("检测到PyTorch Lightning格式的.ckpt文件")
        state_dict = ckpt['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('models.'):
                new_key = key[6:]
            elif key.startswith('net.'):
                new_key = key[4:]
            elif key.startswith('model.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        state_dict_to_load = new_state_dict

    elif isinstance(ckpt, dict) and any(key.startswith(('encoder', 'decoder', 'backbone')) for key in ckpt.keys()):
        logger.info("检测到普通PyTorch格式的.pt文件")
        state_dict_to_load = ckpt

    elif isinstance(ckpt, dict) and all(isinstance(key, str) for key in ckpt.keys()):
        logger.info("检测到直接的state_dict格式")
        state_dict_to_load = ckpt
    else:
        logger.warning("无法识别权重文件格式，尝试直接加载")
        state_dict_to_load = ckpt

    if state_dict_to_load is not None:
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=True)
            logger.info("模型权重严格加载成功")
        except RuntimeError as e:
            logger.warning(f"严格加载失败: {e}，尝试非严格加载")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)
            logger.info("模型权重非严格加载成功")

        if missing_keys:
            logger.warning(f"缺失的键: {len(missing_keys)}个")
        if unexpected_keys:
            logger.warning(f"意外的键: {len(unexpected_keys)}个")
    else:
        logger.error("无法提取有效的state_dict")
        raise ValueError("权重文件格式不支持")

    logger.info("模型加载成功。")
    return model


def get_paths(cfg):
    """
    获取输入图像和掩码路径，增强的匹配逻辑
    支持 .img.nii.gz -> .label.nii.gz 等模式
    """
    image_dir = Path(hydra.utils.to_absolute_path(cfg.image_path))
    # 获取图像文件列表
    pattern = f"*{cfg.image_file_ending}" if cfg.image_file_ending else "*"
    image_paths = sorted(list(image_dir.glob(pattern)))

    logger.info(f"Found {len(image_paths)} image files in {cfg.image_path}")

    if not image_paths:
        raise FileNotFoundError(f"在 {cfg.image_path} 中没有找到任何图像文件")

    # 如果没有指定mask路径，直接返回图像列表（纯推理模式）
    if not cfg.mask_path:
        logger.info("运行在纯推理模式（无标签评估）")
        return image_paths, None

    mask_dir = Path(hydra.utils.to_absolute_path(cfg.mask_path))
    matched_pairs = []

    # 定义常见的图像和标签标记
    # img_tags会被移除以获取ID，label_tags会被添加到ID后以寻找标签
    img_tags_to_strip = ['.img', '_img', '.image', '_image', '_vol', '.vol']
    possible_label_tags = ['.label', '_label', '_seg', '.mask', '_mask', '.seg', '_gt', '.gt']

    for img_path in image_paths:
        filename = img_path.name

        # 1. 智能提取后缀（处理 .nii.gz 双重后缀）
        if filename.endswith('.nii.gz'):
            ext = '.nii.gz'
            base_name = filename[:-7] # 去掉 .nii.gz
        elif filename.endswith('.tar.gz'):
            ext = '.tar.gz'
            base_name = filename[:-7]
        else:
            # 标准单后缀处理
            ext = img_path.suffix
            base_name = img_path.stem

        # 2. 提取核心ID (Core ID)
        # 尝试移除图像特定的标识符，例如 "100.img" -> "100"
        core_id = base_name
        for tag in img_tags_to_strip:
            if base_name.endswith(tag):
                core_id = base_name[:-len(tag)]
                break # 找到一个匹配就停止

        # 3. 构建可能的Mask文件名列表
        candidate_mask_names = []

        # 策略 A: ID + Label Tag + Ext (例如: 100.label.nii.gz)
        for tag in possible_label_tags:
            candidate_mask_names.append(f"{core_id}{tag}{ext}")

        # 策略 B: ID + Ext (同名文件，但在mask文件夹中)
        candidate_mask_names.append(f"{core_id}{ext}")

        # 策略 C: 原始文件名 (例如: 100.img.nii.gz -> 100.img.nii.gz)
        candidate_mask_names.append(filename)

        # 策略 D: 简单的字符串替换 (例如替换 _img 为 _label)
        for i_tag in img_tags_to_strip:
            if i_tag in filename:
                for l_tag in possible_label_tags:
                    candidate_mask_names.append(filename.replace(i_tag, l_tag))

        # 4. 在Mask目录中查找匹配
        found_mask_path = None
        for mask_name in candidate_mask_names:
            potential_path = mask_dir / mask_name
            if potential_path.exists():
                found_mask_path = potential_path
                break

        if found_mask_path:
            matched_pairs.append((img_path, found_mask_path))
            logger.debug(f"Matched: {img_path.name} <-> {found_mask_path.name}")
        else:
            logger.warning(f"No matching mask found for image: {img_path.name} (Core ID: {core_id})")
            if cfg.get('strict_matching', False): # 使用 get 防止配置缺失报错
                raise FileNotFoundError(f"找不到对应的标签文件: {img_path.name}")

    if not matched_pairs:
        if cfg.mask_path:
             logger.error("未找到任何匹配的 图像-标签 对。请检查文件名规则。")
             logger.error(f"示例尝试匹配 ID: {core_id}, 尝试的标签名: {candidate_mask_names[:3]}...")
        return image_paths, None

    # 分离路径
    final_image_paths, final_mask_paths = zip(*matched_pairs)
    logger.info(f"成功匹配 {len(final_image_paths)} 对图像-标签文件")

    return list(final_image_paths), list(final_mask_paths)


def resample(image, factor=None, target_shape=None):
    """
    图像重采样函数
    """
    if factor == 1:
        return image

    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))

    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


def keep_largest_vessels(prediction, num_vessels=3):
    """
    保留预测结果中体积最大的几条血管
    """
    # 确保输入是整数类型且在CPU上
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    prediction = prediction.astype(int)

    # 连通成分分析
    labeled_mask = label(prediction, connectivity=3)
    regions = regionprops(labeled_mask)

    if len(regions) <= num_vessels:
        return prediction

    # 按体积（体素数）排序
    regions_sorted = sorted(regions, key=lambda x: x.area, reverse=True)

    # 创建新的掩码
    processed_mask = np.zeros_like(prediction)

    for i in range(min(num_vessels, len(regions_sorted))):
        region = regions_sorted[i]
        coords = region.coords
        processed_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    return processed_mask


def keep_closest_vessels(prediction, num_vessels=3):
    """
    保留距离图像几何中心最近的几条血管
    Args:
        prediction: 二值分割结果 (numpy数组)
        num_vessels: 要保留的血管数量
    Returns:
        processed_pred: 处理后的二值分割结果
    """
    # 确保输入是整数类型且在CPU上
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    prediction = prediction.astype(int)

    # 连通成分分析
    labeled_mask = label(prediction, connectivity=3)
    regions = regionprops(labeled_mask)

    if len(regions) <= num_vessels:
        return prediction

    # 1. 计算图像中心坐标 (D, H, W) / 2
    image_center = np.array(prediction.shape) / 2.0

    # 2. 定义距离计算函数 (计算连通域质心到图像中心的欧氏距离)
    def get_distance_to_center(region):
        region_centroid = np.array(region.centroid)
        return np.linalg.norm(region_centroid - image_center)

    # 3. 按距离从小到大排序 (最近的排前面)
    regions_sorted = sorted(regions, key=get_distance_to_center)

    # 4. 创建新的掩码，只保留最近的几条血管
    processed_mask = np.zeros_like(prediction)

    for i in range(min(num_vessels, len(regions_sorted))):
        region = regions_sorted[i]
        coords = region.coords
        processed_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    return processed_mask


@hydra.main(config_path="../../configs", config_name="inference", version_base="1.3.2")
def main(cfg):
    """
    推理主函数
    """
    # 设置随机种子
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger.info(f"Using device {cfg.device}.")
    device = torch.device(cfg.device)

    # 加载模型
    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    # 预处理
    transforms = generate_transforms(cfg.transforms_config)

    # 输出目录
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 获取路径
    image_paths, mask_paths = get_paths(cfg)

    # 确定读写器
    file_ending = (cfg.image_file_ending if cfg.image_file_ending else image_paths[0].name)

    # ------------------------------------------------------------
    # 修复点 1: 分离读写器查找后缀 (无点) 和 文件名保存后缀 (有点)
    # ------------------------------------------------------------
    if 'nii.gz' in file_ending or image_paths[0].name.endswith('.nii.gz'):
        rw_suffix = 'nii.gz'      # 用于查找 reader_writer (通常不带点)
        save_ext = '.nii.gz'      # 用于文件名保存 (必须带点)
    else:
        rw_suffix = image_paths[0].suffix
        save_ext = image_paths[0].suffix

    image_reader_writer = determine_reader_writer(rw_suffix)()
    save_writer = determine_reader_writer(rw_suffix)()

    logger.debug(f"Patch size: {cfg.patch_size}, Batch size: {cfg.batch_size}, Overlap: {cfg.overlap}")

    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size,
        sw_batch_size=cfg.batch_size,
        overlap=cfg.overlap,
        mode=cfg.mode,
        sigma_scale=cfg.sigma_scale,
        padding_mode=cfg.padding_mode
    )

    metrics_dict = {}

    with torch.no_grad():
        for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Inference"):
            preds = []

            # 读取图像
            try:
                img_data = image_reader_writer.read_images(image_path)[0].astype(np.float32)
            except Exception as e:
                logger.error(f"读取图像失败 {image_path}: {e}")
                continue

            # TTA 循环
            for scale in cfg.tta.scales:
                # Apply transforms
                image_tensor = transforms(img_data)

                if isinstance(image_tensor, np.ndarray):
                    image_tensor = torch.from_numpy(image_tensor)

                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0) # Add Channel

                image = image_tensor.unsqueeze(0).to(device) # Add Batch

                # Mask 读取 (仅用于 Metric 计算)
                mask = None
                if mask_paths:
                    mask_data = image_reader_writer.read_images(mask_paths[idx])[0]
                    mask = torch.tensor(mask_data).bool()

                # TTA: Invert
                if cfg.tta.invert:
                    if image.mean() > cfg.tta.invert_mean_thresh:
                        image = 1 - image

                # TTA: Hist Eq
                if cfg.tta.equalize_hist:
                    pass

                original_shape = image.shape

                # Resample
                image = resample(image, factor=scale)

                # Inference
                logits = inferer(image, model)

                # Restore shape
                logits = resample(logits, target_shape=original_shape)
                preds.append(logits.cpu().squeeze())

            # Fusion
            if len(preds) > 1:
                stacked_preds = torch.stack(preds)
                if cfg.merging.max:
                    pred = stacked_preds.max(dim=0)[0].sigmoid()
                else:
                    pred = stacked_preds.mean(dim=0).sigmoid()
            else:
                pred = preds[0].sigmoid()

            # Thresholding
            pred_thresh = (pred > cfg.merging.threshold).numpy()

            # Post-processing: Remove small objects
            if cfg.post.apply:
                pred_thresh = remove_small_objects(
                    pred_thresh.astype(bool),
                    min_size=cfg.post.small_objects_min_size,
                    connectivity=cfg.post.small_objects_connectivity
                )

            # Post-processing: Keep largest vessels
            if cfg.post.get('keep_largest_vessels', False):
                pred_thresh = keep_largest_vessels(
                    pred_thresh.astype(int),
                    num_vessels=cfg.post.num_largest_vessels
                )

            # Post-processing: Keep closest vessels
            if cfg.post.get('keep_closest_vessels', False):
                logger.info(f"保留距离中心最近的 {cfg.post.num_closest_vessels} 条血管")
                pred_thresh = keep_closest_vessels(
                    pred_thresh.astype(int),
                    num_vessels=cfg.post.num_closest_vessels
                )

            # ------------------------------------------------------------
            # 修复点 2: 使用 save_ext 确保文件后缀带点
            # ------------------------------------------------------------
            if '.img' in image_path.name:
                clean_name = image_path.name.replace('.img', '')
                if clean_name.endswith('.nii.gz'): clean_name = clean_name[:-7]
                elif clean_name.endswith('.nii'): clean_name = clean_name[:-4]
                # 使用 save_ext 而不是 rw_suffix
                save_name = f"{clean_name}_{cfg.file_app}pred{save_ext}"
            else:
                save_name = f"{image_path.name.split('.')[0]}_{cfg.file_app}pred{save_ext}"

            save_path = output_folder / save_name
            save_writer.write_seg(pred_thresh.astype(np.uint8), save_path)

            # Metrics
            if mask is not None:
                if mask.ndim != pred.ndim:
                    pass

                # --- 修改开始 ---
                # 将后处理后的 numpy 数组转回 tensor，并在 GPU 上计算指标
                # 注意：pred_thresh 已经是 0/1 的二值结果，所以 threshold 参数设为 0.5 即可
                post_processed_tensor = torch.from_numpy(pred_thresh).float().to(device)

                # 使用后处理后的 tensor 进行评估
                metrics = Evaluator().estimate_metrics(post_processed_tensor, mask, threshold=0.5)
                # --- 修改结束 ---

                fname = image_path.name
                logger.info(f"Dice of {fname}: {metrics['dice'].item():.4f}")
                logger.info(f"clDice of {fname}: {metrics['cldice'].item():.4f}")
                metrics_dict[fname] = metrics

    if mask_paths and metrics_dict:
        mean_metrics = calculate_mean_metrics(list(metrics_dict.values()), round_to=cfg.round_to)
        logger.info(f"Mean metrics: dice {mean_metrics['dice'].item():.4f}, cldice {mean_metrics['cldice'].item():.4f}")

    logger.info("Done.")


if __name__ == "__main__":
    main()