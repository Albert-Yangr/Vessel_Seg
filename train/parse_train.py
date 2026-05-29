import logging
import os
import sys
import warnings
import copy
from pathlib import Path

# =========================================================================
# ✅ 路径修复
# =========================================================================
sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import hydra
import torch
from omegaconf import OmegaConf, open_dict, DictConfig, ListConfig

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

# 引用通用业务组件
from utils.evaluation import Evaluator
from utils.dual_branch.dataset import DualStreamDataset, UnlabeledWeakDataset, UnionDataset, MutilSupervisionDataset
from utils.dual_cl.module import DualBranchPLModule

# 🌟 引用 Parse 肺部专属模块 (包含基于面积 400 过滤的逻辑)
from utils.dual_cl.model import DualStreamDynUNet
from utils.dual_cl.loss_parse_com import DualBranchLoss

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class CleanCSVLogger(CSVLogger):
    def log_hyperparams(self, params): pass


class LogCallback(LearningRateMonitor):
    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0: return
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        step = trainer.global_step
        score = m.get("val_DiceMetric")
        best = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None

        logger.info(f"{'=' * 15} Epoch {epoch} | Step {step} {'=' * 15}")
        if score is not None: logger.info(f"✅ 当前验证 Dice: {score:.4f}")
        if best is not None and score is not None:
            diff = float(score) - float(best)
            icon = f"🚀 新纪录! (+{diff:.4f})" if diff > 1e-6 else "⚖️ 持平" if diff > -1e-6 else f"🔙 差距: {diff:.4f}"
            logger.info(f"⭐ 历史最佳: {best:.4f} | {icon}")
        logger.info("=" * 60)


def safe_load_weights(model, checkpoint_path, rank=0):
    if not checkpoint_path or not os.path.exists(checkpoint_path): return
    if rank == 0: logger.info(f"Loading base weights: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    new_state = {f"base_model.{k.replace('model.', '').replace('net.', '')}": v for k, v in state.items()
                 if f"base_model.{k.replace('model.', '').replace('net.', '')}" in model_state}
    model.load_state_dict(new_state, strict=False)
    if rank == 0: logger.info(f"✅ 已加载 {len(new_state)} 个张量至 base_model.")


@hydra.main(config_path="../configs", config_name="train/parse_train", version_base="1.3")
def main(cfg):
    seed_everything(cfg.get("seed", 43), True)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    dataset_name = list(cfg.data.keys())[0]
    run_name = str(cfg.loss_name)
    experiment_dir = f"{cfg.chkpt_folder}/{cfg.data_name}/{run_name}"

    if rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)
        config_save_path = os.path.join(experiment_dir, "train_config.yaml")
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=False))
        logger.info("\n" + "=" * 60)
        logger.info("🫁 启动 Parse 肺部专版：[面积动态剥离] 形态感知主干隔离对比学习")
        logger.info(f"💾 完整配置已备份至: {config_save_path}")
        logger.info("=" * 60)

    resolved_data = OmegaConf.to_container(cfg.data, resolve=True)
    labeled_configs = OmegaConf.create(copy.deepcopy(resolved_data))
    unlabeled_configs = OmegaConf.create(copy.deepcopy(resolved_data))
    val_configs = OmegaConf.create(copy.deepcopy(resolved_data))

    with open_dict(labeled_configs), open_dict(unlabeled_configs), open_dict(val_configs):
        labeled_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, "train")
        unlabeled_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, "train-all")
        val_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path)

    train_ds = DualStreamDataset(
        MutilSupervisionDataset(labeled_configs, mode="train", repeats=cfg.repeats, label_suffix=cfg.label_suffix),
        UnlabeledWeakDataset(unlabeled_configs, mode="train", repeats=cfg.repeats),
    )
    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_ds, shuffle=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader, dataset=UnionDataset(val_configs, mode="test", finetune=True),
                                         batch_size=1)

    model = DualStreamDynUNet(hydra.utils.instantiate(cfg.model))
    safe_load_weights(model, cfg.path_to_chkpt, rank)

    target_loss = cfg.loss_configs.slice_loss if ".slice" in cfg.label_suffix else cfg.loss_configs.label_loss
    cl_cfg = cfg.loss_configs.get("contrastive", None)

    dual_loss = DualBranchLoss(
        hydra.utils.instantiate(target_loss),
        hydra.utils.instantiate(cfg.loss_configs.pseudo_loss),
        cl_cfg=cl_cfg, ramp_epochs=cfg.ramp_epochs, max_pseudo_weight=cfg.pseudo_weight,
    )
    pl_module = DualBranchPLModule(model, dual_loss, Evaluator(), cfg.data_name, cfg.optimizer)

    if rank == 0:
        logger.info("\n" + "✨" * 30)
        logger.info(" 🫁 [Parse 肺部专属训练配置概览]")
        logger.info(f" 🏷️  标签后缀   : {cfg.label_suffix} (请确保包含离线 2D 面积切分标签)")
        if cl_cfg and cl_cfg.get("enable", False):
            logger.info(" ⚔️  肺部面积感知主干隔离 CL : [🟩 启用]")
            logger.info(
                f"    ├── 面积隔离阈值       : {cl_cfg.get('area_threshold')} (🌟 无标签面上大于该面积的血管被忽略)")
            logger.info(f"    ├── 整体损失权重       : {cl_cfg.get('weight')}")
            logger.info(f"    └── 预热轮数 (Warmup)  : {cl_cfg.get('warmup_epochs')}")
        logger.info("✨" * 30 + "\n")

    ckpt_cb = ModelCheckpoint(dirpath=experiment_dir, monitor="val_DiceMetric", mode="max", save_last=True,
                              filename="Epoch{epoch:02d}-{val_DiceMetric:.4f}", save_top_k=1,
                              auto_insert_metric_name=False)
    ckpt_cb.CHECKPOINT_NAME_LAST = f"{run_name}_last"

    devices = cfg.trainer.lightning_trainer.get("devices", [1])
    num_devices = len(devices) if isinstance(devices, (list, tuple, ListConfig)) else int(devices)
    strategy_opt = DDPStrategy(find_unused_parameters=False) if num_devices > 1 else "auto"

    trainer = hydra.utils.instantiate(
        cfg.trainer.lightning_trainer,
        logger=[CleanCSVLogger(save_dir=experiment_dir, name="", version="")],
        callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
        strategy=strategy_opt, sync_batchnorm=(num_devices > 1),
        val_check_interval=cfg.val_frequency, num_sanity_val_steps=0,
    )()

    resume_path = cfg.get("resume_ckpt_path", None)
    if resume_path and os.path.exists(resume_path):
        if rank == 0: logger.info(f"🔄 从断点恢复训练: {resume_path}")
        if torch.distributed.is_initialized(): torch.distributed.barrier()
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=resume_path)
    else:
        trainer.fit(pl_module, train_loader, val_loader)


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()