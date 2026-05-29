import logging, warnings, math, os, csv, re, sys
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
from omegaconf import OmegaConf

# --- 环境设置 ---
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.dataset import generate_transforms
from utils.io import determine_reader_writer
from utils.evaluation import Evaluator, calculate_mean_metrics
from utils.LPMoE.lpmoe_unet import LPMoE_VesselNet

# 屏蔽警告
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("monai").setLevel(logging.ERROR)

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
logger = logging.getLogger(__name__)


# --- 辅助函数 ---
def resample(img, factor=1, target_shape=None):
    if factor == 1 and target_shape is None: return img
    size = target_shape[-3:] if target_shape else [int(round(s / factor)) for s in img.shape[-3:]]
    return F.interpolate(img, size=size, mode="trilinear", align_corners=False)


def apply_post_processing(pred, cfg):
    """整合后的后处理逻辑"""
    if not cfg.post.apply: return pred.astype(int)

    # 1. 去除小物体
    mask = remove_small_objects(pred.astype(bool), min_size=cfg.post.small_objects_min_size)

    if not (cfg.post.get('keep_largest_vessels') or cfg.post.get('keep_closest_vessels')):
        return mask.astype(int)

    # 2. 连通域筛选
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
    """保存 CSV 报告"""
    ts = datetime.now().strftime("%m%d_%H%M")
    path = Path(
        hydra.utils.get_original_cwd()) /  "LPMoE_test" / d_name / f"{cfg.shot_name}shot_{d_name}_{ts}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(metrics.keys())
    headers = list(metrics[keys[0]].keys()) if keys else []
    priority = ['dice', 'cldice', 'clDice', 'iou']
    headers.sort(key=lambda x: (priority.index(x) if x in priority else 99, x))

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerows([
            ["### Config (LPMoE) ###"], ["Time", ts], ["Dataset", d_name], ["Model", cfg.ckpt_path],
            ["Patch/Batch", f"{cfg.patch_size}/{cfg.batch_size}"], []
        ])
        if keys:
            w.writerow(["### Details ###"])
            w.writerow(["Case"] + headers)
            w.writerows([[k] + [metrics[k].get(h, "") for h in headers] for k in keys])
        w.writerows([[], ["### Summary ###"], ["Metric"] + list(mean.keys()), ["Avg"] + [mean[k] for k in mean]])
    logger.info(f"✅ Report saved: {path}")


# --- 核心逻辑 ---
def load_model(cfg, device):
    """专门加载 LPMoE 模型"""
    base_cfg = Path(__file__).resolve().parent.parent / "configs" / "model" / "dyn_unet_base.yaml"
    if not base_cfg.exists(): raise FileNotFoundError(f"Config not found: {base_cfg}")

    model = LPMoE_VesselNet(base_config=OmegaConf.load(base_cfg), num_classes=1).to(device)

    if cfg.ckpt_path and str(cfg.ckpt_path).lower() != "none":
        # weights_only=False 允许加载包含 OmegaConf 的旧版权重
        ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        # 清洗前缀
        state = {k.replace('model.', '').replace('models.', '').replace('net.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        logger.warning("⚠️ Using random init weights!")
    return model.eval()


def worker(rank, gpu, files, masks, cfg, out_dict):
    """单个 GPU 工作进程"""
    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(cfg.seed + rank)
    try:
        model = load_model(cfg, device)
    except Exception as e:
        return print(f"[GPU {gpu}] Model load fail: {e}")

    inferer = SlidingWindowInfererAdapt(roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap,
                                        mode=cfg.mode)
    trans = generate_transforms(cfg.transforms_config)

    # I/O 初始化 (已修复后缀问题)
    fname = files[0].name
    if 'nii.gz' in fname:
        suffix = '.nii.gz'
    else:
        suffix = files[0].suffix

    rw = determine_reader_writer(suffix.lstrip('.'))()

    out_dir = Path(cfg.output_folder) if cfg.output_folder and str(cfg.output_folder).lower() != "none" else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)

    local_res = {}
    pbar = tqdm(zip(files, masks if masks else [None] * len(files)), total=len(files), desc=f"GPU {gpu}", position=rank)

    with torch.no_grad():
        for img_p, msk_p in pbar:
            try:
                img = rw.read_images(img_p)[0].astype(np.float32)
            except:
                continue

            # TTA 推理循环
            preds = []
            for s in cfg.tta.scales:
                data_in = trans(img)
                x = torch.from_numpy(data_in) if isinstance(data_in, np.ndarray) else data_in
                if x.ndim == 3:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.ndim == 4:
                    x = x.unsqueeze(0)
                x = x.to(device)

                if cfg.tta.invert and x.mean() > cfg.tta.invert_mean_thresh: x = 1 - x
                orig_sh = x.shape

                with torch.cuda.amp.autocast(enabled=True):
                    logit = resample(inferer(resample(x, s), model), target_shape=orig_sh)
                preds.append(logit.cpu().squeeze().sigmoid())

            # 融合
            if len(preds) > 1:
                pred = torch.stack(preds).max(dim=0)[0] if cfg.merging.max else torch.stack(preds).mean(dim=0)
            else:
                pred = preds[0]

            # 后处理
            res = apply_post_processing((pred.numpy() > cfg.merging.threshold).astype(int), cfg)

            # 保存预测
            if out_dir:
                clean_name = img_p.name.replace('.img', '').replace('.nii.gz', '').replace(suffix, '')
                save_name = f"{clean_name}_{cfg.file_app}pred{suffix}"
                rw.write_seg(res.astype(np.uint8), out_dir / save_name)

            # 评估指标
            if msk_p:
                m_ts = torch.tensor(rw.read_images(msk_p)[0]).bool().to(device)
                met = Evaluator().estimate_metrics(torch.from_numpy(res).float().to(device), m_ts, threshold=0.5)
                local_res[img_p.name] = {k: v.item() if hasattr(v, 'item') else v for k, v in met.items()}

                # --- 【核心修复：增加 clDice 打印】 ---
                dice = local_res[img_p.name].get('dice', 0)
                cldice = local_res[img_p.name].get('cldice', local_res[img_p.name].get('clDice', 0))
                pbar.write(f"[GPU {gpu}] {img_p.name} | Dice: {dice:.4f} | clDice: {cldice:.4f}")
                # -----------------------------------

    out_dict[rank] = local_res


def resolve_paths(cfg):
    """路径解析 - 包含自动推导逻辑"""
    if str(cfg.image_path).lower() in ["auto", "none"] or str(cfg.output_folder).lower() in ["auto", "none"]:
        try:
            parts = Path(cfg.ckpt_path).parts
            idx = parts.index("checkpoints")
            proj, ds_rel = Path(*parts[:idx - 1]), Path(*parts[idx + 1:-2])
            shot = re.search(r'(\d+)shot', parts[-2]).group(1) if re.search(r'(\d+)shot', parts[-2]) else "0"

            if str(cfg.image_path).lower() in ["auto", "none"]:
                cfg.image_path = str(proj / "datasets" / ds_rel / "test")
            if str(cfg.output_folder).lower() in ["auto", "none"]:
                cfg.output_folder = str(proj / "local_results/output" / ds_rel / f"{shot}_shot_test")
            cfg.shot_name = shot
            logger.info(f"⚡ Auto Paths: Img={cfg.image_path} | Out={cfg.output_folder}")
        except Exception:
            logger.warning("⚠️ Auto path failed, trying original path.")

    root = Path(hydra.utils.to_absolute_path(cfg.image_path))
    if not root.exists(): raise FileNotFoundError(f"Missing: {root}")

    imgs = sorted(list(root.rglob("*.img.nii.gz"))) or sorted(
        [p for p in root.rglob("*.nii.gz") if "label" not in p.name])
    if not imgs: raise FileNotFoundError(f"No images in {root}")

    masks = None
    if cfg.get('mask_suffix'):
        masks = []
        for p in imgs:
            m = p.parent / f"{p.name.split('.img')[0]}{cfg.mask_suffix}"
            if m.exists(): masks.append(m)
        if len(masks) != len(imgs):
            if cfg.get('strict_matching', True): raise ValueError("Mask mismatch")
            masks = None

    return cfg, imgs, masks, root.parent.name


@hydra.main(config_path="../configs", config_name="inference/tem_infer", version_base="1.3.2")
def main(cfg):
    cfg, all_imgs, all_masks, ds_name = resolve_paths(cfg)
    gpus = list(cfg.gpus) if cfg.get("gpus") else [0]

    splits = lambda l, n: [l[i:i + math.ceil(len(l) / n)] for i in range(0, len(l), math.ceil(len(l) / n))]
    chunks_i, chunks_m = splits(all_imgs, len(gpus)), splits(all_masks, len(gpus)) if all_masks else [None] * len(gpus)

    logger.info(f"🚀 Start LPMoE: {len(all_imgs)} files | {len(gpus)} GPUs | Dataset: {ds_name}")

    with mp.Manager() as manager:
        ret = manager.dict()
        procs = [mp.Process(target=worker, args=(i, gpus[i], chunks_i[i], chunks_m[i], cfg, ret)) for i in
                 range(len(chunks_i))]
        [p.start() for p in procs];
        [p.join() for p in procs]

        final = {k: v for d in ret.values() for k, v in d.items()}
        if final:
            means = calculate_mean_metrics(list(final.values()), round_to=cfg.round_to)
            logger.info("\n" + "\n".join([f"Mean {k:<15}: {v:.4f}" for k, v in means.items()]))
            save_report(final, means, cfg, ds_name)
        else:
            logger.info("No metrics.")


if __name__ == "__main__":
    main()