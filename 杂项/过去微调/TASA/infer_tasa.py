import logging, warnings, math, os, csv, re, sys

# --- 【核心修复】强力屏蔽警告 (必须放在其他 import 之前) ---
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
os.environ["PYTHONWARNINGS"] = "ignore"
# ----------------------------------------------------
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import hydra
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

from utils.dataset import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics

# 🌟 引入我们的 TASA 架构
from utils.TASA.tasa_unet import TASA_VesselNet

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
logger = logging.getLogger(__name__)


# =======================================================
# 🚀 后处理模块 (安全字典读取版，免疫 Config 缺失报错)
# =======================================================
def resample(img, factor=1, target_shape=None):
    if factor == 1 and target_shape is None: return img
    size = target_shape[-3:] if target_shape else [int(round(s / factor)) for s in img.shape[-3:]]
    return F.interpolate(img, size=size, mode="trilinear", align_corners=False)


def apply_post_processing(pred, cfg):
    """整合后的安全后处理逻辑"""
    post_cfg = cfg.get("post", {})
    # 如果没开后处理，直接原样返回
    if not post_cfg or not post_cfg.get("apply", False):
        return pred

    # 1. 移除微小孤立噪点
    min_size = post_cfg.get("small_objects_min_size", 64)
    mask = remove_small_objects(pred.astype(bool), min_size=min_size)

    keep_largest = post_cfg.get('keep_largest_vessels', False)
    keep_closest = post_cfg.get('keep_closest_vessels', False)

    if not (keep_largest or keep_closest):
        return mask.astype(int)

    # 2. 连通域提取与分析
    lbl, num = label(mask, return_num=True, connectivity=3)
    if num == 0:
        return mask.astype(int)
    props = regionprops(lbl)

    if keep_largest:
        n_largest = post_cfg.get('num_largest_vessels', 1)
        targets = sorted(props, key=lambda x: x.area, reverse=True)[:n_largest]
    elif keep_closest:
        n_closest = post_cfg.get('num_closest_vessels', 1)
        center = np.array(pred.shape) / 2.0
        targets = sorted(props, key=lambda x: np.linalg.norm(np.array(x.centroid) - center))[:n_closest]
    else:
        return mask.astype(int)

    out = np.zeros_like(pred)
    for r in targets:
        out[tuple(r.coords.T)] = 1
    return out


# =======================================================
# 报告保存与模型加载
# =======================================================
def save_report(metrics, mean, cfg, d_name):
    """精简版CSV报告生成"""
    ts = datetime.now().strftime("%m%d_%H%M")
    path = Path(hydra.utils.get_original_cwd()) / "few_shot" / d_name / f"{cfg.sign}_{d_name}_{ts}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(metrics.keys())
    headers = list(metrics[keys[0]].keys()) if keys else []

    priority = ['dice', 'cldice', 'clDice', 'nsd', 'asd']
    headers.sort(key=lambda x: (priority.index(x) if x in priority else 99, x))
    mean_keys = sorted(mean.keys(), key=lambda x: (priority.index(x) if x in priority else 99, x))

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)

        # 安全获取显示参数
        scales = cfg.get("tta", {}).get("scales", [1.0])
        invert = cfg.get("tta", {}).get("invert", False)
        post_applied = cfg.get("post", {}).get("apply", False)

        w.writerows([
            ["### Config ###"], ["Time", ts], ["Model", cfg.ckpt_path],
            ["Expert", str(cfg.get("expert_type", "None"))],
            ["TTA", f"{scales} (Inv:{invert})"], ["Post", f"{post_applied}"], []
        ])
        if keys:
            w.writerow(["### Details ###"])
            w.writerow(["Case"] + headers)
            w.writerows([[k] + [metrics[k].get(h, "") for h in headers] for k in keys])

        w.writerows([[], ["### Summary ###"], ["Metric"] + mean_keys, ["Avg"] + [mean[k] for k in mean_keys]])
    logger.info(f"✅ Report saved: {path}")


def load_model(cfg, device):
    """
    专门适配 TASA 寄生架构的模型权重加载逻辑
    """
    expert_type = cfg.get("expert_type", "snake")
    print(f"🔄 正在构建 TASA 寄生推理架构，挂载专家模块: {expert_type}")

    # 1. 实例化原汁原味的基础大模型 (由 Hydra 自动解析 yaml)
    base_model = hydra.utils.instantiate(cfg.model)

    # 2. 将基础大模型包裹入 TASA 寄生壳中
    model = TASA_VesselNet(base_model=base_model, expert_type=expert_type).to(device)

    print(f"📂 正在加载权重: {cfg.ckpt_path}")
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    clean_state_dict = {}
    for k, v in state_dict.items():
        if "projection_head" in k or "loss" in k:
            continue
        # 清洗 Lightning 自动添加的前缀
        clean_key = k
        if clean_key.startswith("model."):
            clean_key = clean_key.replace("model.", "", 1)
        elif clean_key.startswith("net."):
            clean_key = clean_key.replace("net.", "", 1)
        clean_state_dict[clean_key] = v

    try:
        model.load_state_dict(clean_state_dict, strict=True)
        print("✅ 权重加载成功 (Strict Mode: 100% 完美贴合)")
    except RuntimeError as e:
        print("⚠️ 权重加载不完全匹配，退化为 Strict=False。报错信息如下：")
        print(e)
        model.load_state_dict(clean_state_dict, strict=False)

    return model.eval()


def worker(rank, gpu, files, masks, cfg, out_dict):
    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(cfg.seed + rank)
    try:
        model = load_model(cfg, device)
    except Exception as e:
        return print(f"[GPU {gpu}] Model load fail: {e}")

    # 获取安全参数
    inferer_mode = cfg.get("mode", "gaussian")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size,
        sw_batch_size=cfg.batch_size,
        overlap=cfg.overlap,
        mode=inferer_mode
    )

    # 预处理数据增强加载兜底
    trans_cfg = cfg.get("transforms_config", [])
    if trans_cfg:
        trans = generate_transforms(trans_cfg)
    else:
        # 如果配置文件未提供，兜底方法：直接返回原数据
        trans = lambda x: x

    fname = files[0].name
    suffix = 'nii.gz' if 'nii.gz' in fname else files[0].suffix
    try:
        io_cls = determine_reader_writer(suffix)
    except ValueError:
        io_cls = determine_reader_writer(suffix.strip('.'))

    rw = io_cls()
    output_dir = Path(cfg.output_folder) if cfg.output_folder and str(cfg.output_folder).lower() != "none" else None
    if output_dir: output_dir.mkdir(parents=True, exist_ok=True)

    local_res = {}
    pbar = tqdm(zip(files, masks if masks else [None] * len(files)), total=len(files), desc=f"GPU {gpu}", position=rank)

    with torch.no_grad():
        for img_p, msk_p in pbar:
            try:
                img = rw.read_images(img_p)[0].astype(np.float32)
            except Exception as e:
                print(f"Read Error: {img_p} -> {e}")
                continue

            # ==================================
            # 🔄 TTA (测试时增强) 推理循环
            # ==================================
            preds = []
            tta_cfg = cfg.get("tta", {})
            scales = tta_cfg.get("scales", [1.0])
            invert = tta_cfg.get("invert", False)
            invert_thresh = tta_cfg.get("invert_mean_thresh", 0.5)

            for s in scales:
                data_in = trans(img)
                x = torch.from_numpy(data_in) if isinstance(data_in, np.ndarray) else data_in

                if x.ndim == 3:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.ndim == 4:
                    x = x.unsqueeze(0)
                x = x.to(device)

                if invert and x.mean() > invert_thresh:
                    x = 1 - x
                orig_sh = x.shape

                # 滑动窗口推理
                logit = resample(inferer(resample(x, s), model), target_shape=orig_sh)
                preds.append(logit.cpu().squeeze().sigmoid())

            # ==================================
            # 融合与后处理
            # ==================================
            merging_cfg = cfg.get("merging", {})
            use_max = merging_cfg.get("max", True)
            threshold = merging_cfg.get("threshold", 0.5)

            if len(preds) > 1:
                pred = torch.stack(preds).max(dim=0)[0] if use_max else torch.stack(preds).mean(dim=0)
            else:
                pred = preds[0]

            # 🔥 调用刚强化过的安全后处理模块
            res = apply_post_processing((pred.numpy() > threshold).astype(int), cfg)

            if output_dir:
                save_ext = '.nii.gz' if 'nii.gz' in fname else files[0].suffix
                clean_name = img_p.name.replace('.img', '').replace('.nii.gz', '').replace(save_ext, '')
                save_name = f"{clean_name}_{cfg.file_app}pred{save_ext}"
                rw.write_seg(res.astype(np.uint8), output_dir / save_name)

            if cfg.get("evaluate", True) and msk_p:
                m_ts = torch.tensor(rw.read_images(msk_p)[0]).bool().to(device)
                met = Evaluator().estimate_metrics(torch.from_numpy(res).float().to(device), m_ts, threshold=0.5)
                met_v = {k: v.item() if hasattr(v, 'item') else v for k, v in met.items()}
                local_res[img_p.name] = met_v
                pbar.write(
                    f"[GPU {gpu}] {img_p.name} | Dice: {met_v.get('dice', 0):.4f} | clDice: {met_v.get('cldice', met_v.get('clDice', 0)):.4f}")

    out_dict[rank] = local_res


def resolve_paths(cfg):
    """自动路径解析与列表获取"""
    # 1. 自动推导
    if str(cfg.image_path).lower() in ["auto", "none"] or str(cfg.output_folder).lower() in ["auto", "none"]:
        try:
            parts = Path(cfg.ckpt_path).parts
            idx = parts.index("checkpoints")
            proj, ds_rel = Path(*parts[:idx - 1]), Path(*parts[idx + 1:-2])
            sign = cfg.sign
            if str(cfg.image_path).lower() in ["auto", "none"]:
                cfg.image_path = str(proj / "datasets" / ds_rel / "test")
            if str(cfg.output_folder).lower() in ["auto", "none"]:
                cfg.output_folder = str(proj / "local_results/output" / ds_rel / f"{sign}_test")

            logger.info(f"⚡ Auto Paths: Img={cfg.image_path} | Out={cfg.output_folder}")
        except Exception:
            logger.warning("⚠️ Auto path failed, using original.");
            pass

    root = Path(cfg.image_path)
    imgs = sorted([p for p in root.glob("*/*.img.nii.gz")]) or sorted(
        [p for p in root.glob("*/*.nii.gz") if "label" not in p.name])
    if not imgs: raise FileNotFoundError(f"No images in {root}")

    masks = None
    if cfg.get("evaluate", True):
        if cfg.mask_path or cfg.mask_suffix:
            suffix = cfg.mask_suffix or ".label.nii.gz"
            masks = [p.parent / f"{p.name.split('.img')[0]}{suffix}" for p in imgs]
            if not all(m.exists() for m in masks):
                if cfg.get('strict_matching', True): raise ValueError("Missing masks!")
                logger.warning("Masks missing, evaluation disabled.");
                masks = None
    else:
        logger.info("⏩ Evaluation disabled (cfg.evaluate=False), skipping mask loading.")

    return cfg, imgs, masks, root.parent.name


@hydra.main(config_path="../configs", config_name="inference/tasa_infer", version_base="1.3.2")
def main(cfg):
    cfg, all_imgs, all_masks, ds_name = resolve_paths(cfg)
    gpus = list(cfg.gpus) if cfg.get("gpus") else [0]

    splits = lambda l, n: [l[i:i + math.ceil(len(l) / n)] for i in range(0, len(l), math.ceil(len(l) / n))]
    chunks_i, chunks_m = splits(all_imgs, len(gpus)), splits(all_masks, len(gpus)) if all_masks else [None] * len(gpus)

    mode_str = "Evaluation & Inference" if cfg.get("evaluate", True) else "Inference Only (Fast)"
    logger.info(f"🚀 Start [{mode_str}]: {len(all_imgs)} files | {len(gpus)} GPUs | Dataset: {ds_name}")

    with mp.Manager() as manager:
        ret_dict = manager.dict()
        procs = [mp.Process(target=worker, args=(i, gpus[i], chunks_i[i], chunks_m[i], cfg, ret_dict)) for i in
                 range(len(chunks_i))]
        [p.start() for p in procs];
        [p.join() for p in procs]

        final_res = {k: v for d in ret_dict.values() for k, v in d.items()}

        if final_res and cfg.get("evaluate", True):
            means = calculate_mean_metrics(list(final_res.values()), round_to=cfg.round_to)
            priority = ['dice', 'cldice', 'clDice', 'nsd', 'asd']
            sorted_mean_keys = sorted(means.keys(), key=lambda x: (priority.index(x) if x in priority else 99, x))

            logger.info("\n" + "\n".join([f"Mean {k:<15}: {means[k]:.4f}" for k in sorted_mean_keys]))
            save_report(final_res, means, cfg, ds_name)
        else:
            if not cfg.get("evaluate", True):
                logger.info("✅ Inference completed. (Evaluation skipped)")
            else:
                logger.info("⚠️ No metrics calculated (Check masks or strict_matching).")


if __name__ == "__main__":
    main()