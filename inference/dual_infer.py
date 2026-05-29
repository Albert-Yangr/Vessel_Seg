import logging, warnings, math, os, csv
from pathlib import Path
from datetime import datetime

# --- 【核心修复】强力屏蔽警告 ---
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
os.environ["PYTHONWARNINGS"] = "ignore"
# ----------------------------------------------------

import torch
import torch.multiprocessing as mp
import hydra
import numpy as np
from tqdm import tqdm

# 核心：直接使用与 module.py 中 validation_step 完全一致的接口
from monai.inferers import sliding_window_inference

# === 新增：引入 skimage 相关的形态学后处理库 ===
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

from utils.dataset import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)


# === 新增：后处理函数 (与 infer_test.py 保持一致，受 cfg.post 控制) ===
def apply_post_processing(pred, cfg):
    """整合后的后处理逻辑"""
    if getattr(cfg, "post", None) is None or not cfg.post.get("apply", False):
        return pred

    # 去除小物体 (噪点)
    mask = remove_small_objects(pred.astype(bool), min_size=cfg.post.small_objects_min_size)

    if not (cfg.post.get('keep_largest_vessels') or cfg.post.get('keep_closest_vessels')):
        return mask.astype(int)

    # 连通域筛选 (最大 or 最近)
    lbl, num = label(mask, return_num=True, connectivity=3)
    if num == 0: return mask.astype(int)
    props = regionprops(lbl)

    if cfg.post.get('keep_largest_vessels'):
        targets = sorted(props, key=lambda x: x.area, reverse=True)[:cfg.post.num_largest_vessels]
    elif cfg.post.get('keep_closest_vessels'):
        center = np.array(pred.shape) / 2.0
        targets = sorted(props, key=lambda x: np.linalg.norm(np.array(x.centroid) - center))[
            :cfg.post.num_closest_vessels]
    else:
        return mask.astype(int)

    out = np.zeros_like(pred)
    for r in targets: out[tuple(r.coords.T)] = 1
    return out


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
        w.writerows([
            ["### Config ###"], ["Time", ts], ["Model", cfg.ckpt_path],
            ["Mode", "Strict Validation Sync (With Optional Post-Processing)"],
            ["Post-Processing", f"Applied: {cfg.post.apply}" if hasattr(cfg, "post") else "None"], []
        ])
        if keys:
            w.writerow(["### Details ###"])
            w.writerow(["Case"] + headers)
            w.writerows([[k] + [metrics[k].get(h, "") for h in headers] for k in keys])

        w.writerows([[], ["### Summary ###"], ["Metric"] + mean_keys, ["Avg"] + [mean[k] for k in mean_keys]])
    logger.info(f"✅ Report saved: {path}")


def load_model(cfg, device):
    """加载模型权重 (仅保留主干网络以匹配验证/推理逻辑)"""
    print(f"🔄 Loading model structure: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model).to(device)

    print(f"📂 Loading weights from: {cfg.ckpt_path}")
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    new_state = {}
    for k, v in state.items():
        if "projection_head" in k: continue

        # 过滤辅分支参数，避免 strict 加载报错
        if "aux_upsamples" in k or "aux_output_block" in k: continue

        new_k = k
        for prefix in ['model.', 'net.', 'module.']:
            if new_k.startswith(prefix): new_k = new_k.replace(prefix, '')
        if new_k.startswith('base_model.'):
            new_k = new_k.replace('base_model.', '')
        elif new_k.startswith('base_'):
            new_k = new_k.replace('base_', '')
        new_state[new_k] = v

    try:
        model.load_state_dict(new_state, strict=True)
        print("✅ 权重加载成功 (Strict Mode)")
    except RuntimeError as e:
        print(f"⚠️ 权重加载不完全匹配 (Strict=False Mode)\n   {str(e)[:200]}...")
        model.load_state_dict(new_state, strict=False)

    return model.eval()


def worker(rank, gpu, files, masks, cfg, out_dict):
    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(cfg.seed + rank)
    try:
        model = load_model(cfg, device)
    except Exception as e:
        return print(f"[GPU {gpu}] Model load fail: {e}")

    trans = generate_transforms(cfg.transforms_config)

    fname = files[0].name
    suffix = 'nii.gz' if 'nii.gz' in fname else files[0].suffix
    try:
        io_cls = determine_reader_writer(suffix)
    except ValueError:
        io_cls = determine_reader_writer(suffix.strip('.'))

    rw = io_cls()
    output_dir = Path(cfg.output_folder) if cfg.output_folder and cfg.output_folder.lower() != "none" else None
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

            # 1. 预处理
            data_in = trans(img)
            x = torch.from_numpy(data_in) if isinstance(data_in, np.ndarray) else data_in

            if x.ndim == 3:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.ndim == 4:
                x = x.unsqueeze(0)
            x = x.to(device)

            # 2. 推理 (100% 对齐 module.py)
            val_outputs = sliding_window_inference(
                inputs=x,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )

            # sigmoid 激活并以 0.5 作为阈值二值化
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()

            # 转为 numpy
            pred_np = val_preds.cpu().squeeze().numpy().astype(np.uint8)

            # ==============================================================
            # 🌟 新增：执行形态学后处理 (去除噪点等)
            # ==============================================================
            pred_np = apply_post_processing(pred_np, cfg).astype(np.uint8)

            # 3. 结果保存
            if output_dir:
                save_ext = '.nii.gz' if 'nii.gz' in fname else files[0].suffix
                clean_name = img_p.name.replace('.img', '').replace('.nii.gz', '').replace(save_ext, '')
                save_name = f"{clean_name}_{cfg.file_app}pred{save_ext}"
                rw.write_seg(pred_np, output_dir / save_name)

            # 4. 评估计算
            if cfg.get("evaluate", True) and msk_p:
                m_ts = torch.tensor(rw.read_images(msk_p)[0]).bool().to(device)

                met = Evaluator().estimate_metrics(torch.from_numpy(pred_np).float().to(device), m_ts, threshold=0.5)
                met_v = {k: v.item() if hasattr(v, 'item') else v for k, v in met.items()}

                local_res[img_p.name] = met_v
                pbar.write(
                    f"[GPU {gpu}] {img_p.name} | Dice: {met_v.get('dice', 0):.4f} | clDice: {met_v.get('cldice', met_v.get('clDice', 0)):.4f}")

    out_dict[rank] = local_res


def resolve_paths(cfg):
    """自动路径解析 (保持不变)"""
    if str(cfg.image_path).lower() in ["auto", "none"] or str(cfg.output_folder).lower() in ["auto", "none"]:
        try:
            parts = Path(cfg.ckpt_path).parts
            idx = parts.index("checkpoints")
            proj, ds_rel = Path(*parts[:idx - 1]), Path(*parts[idx + 1:-2])
            sign = cfg.sign
            if str(cfg.image_path).lower() in ["auto", "none"]:
                cfg.image_path = str(proj / "datasets" / ds_rel / "test")
            if str(cfg.output_folder).lower() in ["auto", "none"]:
                cfg.output_folder = str(proj / "local_results/output" / ds_rel / f"{sign}")
            logger.info(f"⚡ Auto Paths: Img={cfg.image_path} | Out={cfg.output_folder}")
        except Exception:
            logger.warning("⚠️ Auto path failed, using original.")
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
                logger.warning("Masks missing, evaluation disabled.")
                masks = None
    else:
        logger.info("⏩ Evaluation disabled (cfg.evaluate=False), skipping mask loading.")

    return cfg, imgs, masks, root.parent.name


@hydra.main(config_path="../configs", config_name="inference/tem_infer", version_base="1.3.2")
def main(cfg):
    cfg, all_imgs, all_masks, ds_name = resolve_paths(cfg)
    gpus = list(cfg.gpus) if cfg.get("gpus") else [0]

    splits = lambda l, n: [l[i:i + math.ceil(len(l) / n)] for i in range(0, len(l), math.ceil(len(l) / n))]
    chunks_i, chunks_m = splits(all_imgs, len(gpus)), splits(all_masks, len(gpus)) if all_masks else [None] * len(gpus)

    mode_str = "Strict Sync Validation & Inference" if cfg.get("evaluate", True) else "Inference Only"
    logger.info(f"🚀 Start [{mode_str}]: {len(all_imgs)} files | {len(gpus)} GPUs | Dataset: {ds_name}")

    with mp.Manager() as manager:
        ret_dict = manager.dict()
        procs = [mp.Process(target=worker, args=(i, gpus[i], chunks_i[i], chunks_m[i], cfg, ret_dict)) for i in
                 range(len(chunks_i))]
        [p.start() for p in procs]
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
                logger.info("✅ Inference completed.")
            else:
                logger.info("⚠️ No metrics calculated.")


if __name__ == "__main__":
    main()