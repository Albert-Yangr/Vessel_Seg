import logging, warnings, math, os, csv, re, sys

# --- 【核心修复】强力屏蔽警告 (必须放在其他 import 之前) ---
warnings.filterwarnings("ignore")
# 专门针对 monai/pkg_resources 的顽固警告
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
# 确保多进程(mp.spawn)启动的子进程也能屏蔽警告
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

warnings.filterwarnings("ignore")
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
logger = logging.getLogger(__name__)


# --- 辅助函数：后处理与图像操作 ---
def resample(img, factor=1, target_shape=None):
    if factor == 1 and target_shape is None: return img
    size = target_shape[-3:] if target_shape else [int(round(s / factor)) for s in img.shape[-3:]]
    return F.interpolate(img, size=size, mode="trilinear", align_corners=False)


def apply_post_processing(pred, cfg):
    """整合后的后处理逻辑"""
    if not cfg.post.apply: return pred
    # 去除小物体
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
    path = Path(
        hydra.utils.get_original_cwd()) / "few_shot" / d_name / f"{cfg.sign}_{d_name}_{ts}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(metrics.keys())
    headers = list(metrics[keys[0]].keys()) if keys else []

    # 定义排序优先级：Dice -> clDice -> NSD -> ASD -> 其他
    priority = ['dice', 'cldice', 'clDice', 'nsd', 'asd']

    # 1. 排序 Detailed 表头
    headers.sort(key=lambda x: (priority.index(x) if x in priority else 99, x))

    # 2. 排序 Summary 键
    mean_keys = sorted(mean.keys(), key=lambda x: (priority.index(x) if x in priority else 99, x))

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerows([
            ["### Config ###"], ["Time", ts], ["Model", cfg.ckpt_path],
            ["TTA", f"{cfg.tta.scales} (Inv:{cfg.tta.invert})"], ["Post", f"{cfg.post.apply}"], []
        ])
        if keys:
            w.writerow(["### Details ###"])
            w.writerow(["Case"] + headers)
            w.writerows([[k] + [metrics[k].get(h, "") for h in headers] for k in keys])

        # 使用排序后的 mean_keys 写入 Summary
        w.writerows([[], ["### Summary ###"], ["Metric"] + mean_keys, ["Avg"] + [mean[k] for k in mean_keys]])
    logger.info(f"✅ Report saved: {path}")


def load_model(cfg, device):
    """
    加载模型权重（增强版：自动适配各种前缀）
    """
    print(f"🔄 Loading model structure: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model).to(device)

    print(f"📂 Loading weights from: {cfg.ckpt_path}")
    # 显式添加 weights_only=False 以兼容旧版
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)

    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    # === 核心修复：更暴力的前缀清洗 ===
    new_state = {}
    for k, v in state.items():
        # 1. 过滤掉对比学习的投影头 (推理不需要)
        if "projection_head" in k:
            continue

        # 2. 逐步清洗前缀
        new_k = k
        # 常见包装器前缀
        for prefix in ['model.', 'net.', 'module.']:
            if new_k.startswith(prefix):
                new_k = new_k.replace(prefix, '')

        # 对比学习包装器前缀 (base_model. 或 base_)
        if new_k.startswith('base_model.'):
            new_k = new_k.replace('base_model.', '')
        elif new_k.startswith('base_'):  # <--- 针对你当前报错的关键修复
            new_k = new_k.replace('base_', '')

        new_state[new_k] = v

    # 3. 尝试加载
    try:
        # 先尝试严格加载，如果成功则完美
        model.load_state_dict(new_state, strict=True)
        print("✅ 权重加载成功 (Strict Mode)")
    except RuntimeError as e:
        # 如果失败，打印详细信息并尝试非严格加载
        print(f"⚠️ 权重加载不完全匹配 (Strict=False Mode)")
        print(f"   错误信息摘要: {str(e)[:200]}...")  # 只打印前200字符避免刷屏

        # 再次检查是否有些关键层没加载上
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(new_state.keys())
        missing = model_keys - loaded_keys
        if len(missing) > 0:
            print(f"   🔻 警告: 有 {len(missing)} 个层未加载权重 (可能是分类头或新层)")
            # 打印前3个缺失的层名用于调试
            print(f"   Examples: {list(missing)[:3]}")

        model.load_state_dict(new_state, strict=False)

    return model.eval()


def worker(rank, gpu, files, masks, cfg, out_dict):
    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(cfg.seed + rank)
    try:
        model = load_model(cfg, device)
    except Exception as e:
        return print(f"[GPU {gpu}] Model load fail: {e}")

    inferer = SlidingWindowInfererAdapt(roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap,
                                        mode=cfg.mode)
    trans = generate_transforms(cfg.transforms_config)

    # I/O 后缀逻辑
    fname = files[0].name
    suffix = 'nii.gz' if 'nii.gz' in fname else files[0].suffix
    try:
        io_cls = determine_reader_writer(suffix)
    except ValueError:
        io_cls = determine_reader_writer(suffix.strip('.'))

    rw, output_dir = io_cls(), Path(
        cfg.output_folder) if cfg.output_folder and cfg.output_folder.lower() != "none" else None
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

            # TTA 推理
            preds = []
            for s in cfg.tta.scales:
                # 1. 获取增强后的数据
                data_in = trans(img)

                # 2. 如果是 numpy 则转 tensor
                if isinstance(data_in, np.ndarray):
                    x = torch.from_numpy(data_in)
                else:
                    x = data_in  # 已经是 Tensor/MetaTensor

                # 3. 维度修正
                if x.ndim == 3:  # (D, H, W)
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.ndim == 4:  # (C, D, H, W)
                    x = x.unsqueeze(0)

                x = x.to(device)

                if cfg.tta.invert and x.mean() > cfg.tta.invert_mean_thresh: x = 1 - x
                orig_sh = x.shape
                logit = resample(inferer(resample(x, s), model), target_shape=orig_sh)
                preds.append(logit.cpu().squeeze().sigmoid())

            # 融合
            if len(preds) > 1:
                pred = torch.stack(preds).max(dim=0)[0] if cfg.merging.max else torch.stack(preds).mean(dim=0)
            else:
                pred = preds[0]

            res = apply_post_processing((pred.numpy() > cfg.merging.threshold).astype(int), cfg)

            # 保存
            if output_dir:
                save_ext = '.nii.gz' if 'nii.gz' in fname else files[0].suffix
                clean_name = img_p.name.replace('.img', '').replace('.nii.gz', '').replace(save_ext, '')
                save_name = f"{clean_name}_{cfg.file_app}pred{save_ext}"
                rw.write_seg(res.astype(np.uint8), output_dir / save_name)

            # 【修改点】评估逻辑：增加 evaluate 开关判断
            # 只有当 evaluate=True 且 mask 存在时才计算
            if cfg.get("evaluate", True) and msk_p:
                m_ts = torch.tensor(rw.read_images(msk_p)[0]).bool().to(device)
                met = Evaluator().estimate_metrics(torch.from_numpy(res).float().to(device), m_ts, threshold=0.5)
                met_v = {k: v.item() if hasattr(v, 'item') else v for k, v in met.items()}
                local_res[img_p.name] = met_v
                pbar.write(
                    f"[GPU {gpu}] {img_p.name} | Dice: {met_v.get('dice', 0):.4f} | clDice: {met_v.get('cldice', met_v.get('clDice', 0)):.4f}")
            elif not cfg.get("evaluate", True):
                # 只是为了让进度条动一下，或打印纯推理信息（可选）
                pass

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
                cfg.output_folder = str(proj / "local_results/output" / ds_rel / f"{sign}")

            logger.info(f"⚡ Auto Paths: Img={cfg.image_path} | Out={cfg.output_folder}")
        except Exception:
            logger.warning("⚠️ Auto path failed, using original.");
            pass

    # 2. 获取文件列表
    root = Path(cfg.image_path)
    imgs = sorted([p for p in root.glob("*/*.img.nii.gz")]) or sorted(
        [p for p in root.glob("*/*.nii.gz") if "label" not in p.name])
    if not imgs: raise FileNotFoundError(f"No images in {root}")

    # 【修改点】根据 evaluate 开关决定是否加载 Mask
    masks = None
    if cfg.get("evaluate", True):
        if cfg.mask_path or cfg.mask_suffix:
            suffix = cfg.mask_suffix or ".label.nii.gz"
            masks = [p.parent / f"{p.name.split('.img')[0]}{suffix}" for p in imgs]
            # 只有在需要评估时才检查 mask 是否存在
            if not all(m.exists() for m in masks):
                if cfg.get('strict_matching', True): raise ValueError("Missing masks!")
                logger.warning("Masks missing, evaluation disabled.");
                masks = None
    else:
        logger.info("⏩ Evaluation disabled (cfg.evaluate=False), skipping mask loading.")

    return cfg, imgs, masks, root.parent.name


@hydra.main(config_path="../configs", config_name="inference/tem_infer", version_base="1.3.2")
def main(cfg):
    cfg, all_imgs, all_masks, ds_name = resolve_paths(cfg)
    gpus = list(cfg.gpus) if cfg.get("gpus") else [0]

    # 切分数据
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

        # 只有在有结果时才生成报告
        if final_res and cfg.get("evaluate", True):
            means = calculate_mean_metrics(list(final_res.values()), round_to=cfg.round_to)

            # 控制台打印也按优先级排序
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