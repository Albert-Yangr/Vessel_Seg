import logging
import sys
import warnings
import os
from pathlib import Path

# =========================================================================
# ✅ 路径修复：必须在导入任何自定义 utils 模块之前将根目录加入系统路径
# =========================================================================
sys.path.append(str(Path(__file__).resolve().parent.parent))

# =========================================================================
# 🌟 修复线程爆炸：限制底层 C++ 库的多线程并发
# =========================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import copy
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf, open_dict, DictConfig, ListConfig

# 引入 OmegaConf 并在 PyTorch 中注册安全白名单
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

# ==========================================================
# 🌟 引用新建的 dual_cl 组件
# ==========================================================
from utils.evaluation import Evaluator
from utils.dual_branch.dataset import DualStreamDataset, UnlabeledWeakDataset, UnionDataset, MutilSupervisionDataset
from utils.dual_cl.model import DualStreamDynUNet
from utils.dual_cl.loss import DualBranchLoss
from utils.dual_cl.module import DualBranchPLModule
from utils.experiment_tracker import save_experiment_record

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class CleanCSVLogger(CSVLogger):
    def log_hyperparams(self, params):
        pass


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
    if rank == 0: logger.info(f"正在加载基础权重: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    new_state = {f"base_model.{k.replace('model.', '').replace('net.', '')}": v
                 for k, v in state.items() if
                 f"base_model.{k.replace('model.', '').replace('net.', '')}" in model_state}
    model.load_state_dict(new_state, strict=False)
    if rank == 0: logger.info(f"✅ 已加载基础权重")


# 🌟 使用新的配置文件 dual_cl
@hydra.main(config_path="../configs", config_name="train/dual_cl", version_base="1.3")
def main(cfg):
    seed_everything(cfg.seed, True)
    rank = int(os.environ.get("LOCAL_RANK", 0))

    dataset_name = list(cfg.data.keys())[0]
    run_name = f'{cfg.loss_name}'

    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("🚀 [Dual-Branch CL] 切片自适应对比学习启动")
        logger.info("=" * 60)

    experiment_dir = f"{cfg.chkpt_folder}/{cfg.data_name}/{run_name}"

    if rank == 0:

        os.makedirs(experiment_dir, exist_ok=True)

        # ==========================================================
        # 🌟 新增：把完整的 Hydra 配置保存到当前的 checkpoint 文件夹中
        # ==========================================================
        config_save_path = os.path.join(experiment_dir, "train_config.yaml")
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=False))
        save_experiment_record(cfg, __file__, experiment_dir, logger)
        logger.info(f"💾 完整配置已备份至: {config_save_path}")

    labeled_configs = copy.deepcopy(cfg.data)
    unlabeled_configs = copy.deepcopy(cfg.data)
    val_configs = copy.deepcopy(cfg.data)
    with open_dict(labeled_configs), open_dict(unlabeled_configs), open_dict(val_configs):
        labeled_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, "train")
        unlabeled_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, "train-all")
        val_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path)

    train_ds = DualStreamDataset(
        MutilSupervisionDataset(labeled_configs, mode="train", repeats=cfg.repeats, label_suffix=cfg.label_suffix),
        UnlabeledWeakDataset(unlabeled_configs, mode="train", repeats=cfg.repeats)
    )
    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_ds, shuffle=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader, dataset=UnionDataset(val_configs, mode="test", finetune=True),
                                         batch_size=1)

    model = DualStreamDynUNet(hydra.utils.instantiate(cfg.model))
    safe_load_weights(model, cfg.path_to_chkpt, rank)

    # 🌟 修改点：灵活的字符串匹配，完美兼容 .npy
    target_loss = cfg.loss_configs.slice_loss if ".slice" in cfg.label_suffix else cfg.loss_configs.label_loss

    cl_cfg = cfg.loss_configs.get("contrastive", None)

    dual_loss = DualBranchLoss(
        hydra.utils.instantiate(target_loss),
        hydra.utils.instantiate(cfg.loss_configs.pseudo_loss),
        cl_cfg=cl_cfg,
        ramp_epochs=cfg.ramp_epochs,
        max_pseudo_weight=cfg.pseudo_weight,
        pseudo_label_mode=cfg.get("pseudo_label_mode", "hard")
    )

    pl_module = DualBranchPLModule(model, dual_loss, Evaluator(), cfg.data_name, cfg.optimizer)

    # 🌟 修改点：更新全景日志打印，反映统一尺寸的新逻辑
    if rank == 0:
        logger.info("\n" + "✨" * 40)
        logger.info(" 📋 [训练核心配置概览 - 包含对比学习]")
        logger.info("✨" * 40)
        logger.info(f" 📁 数据集名称 : {cfg.data_name} ({dataset_name})")
        logger.info(f" 📂 有标签路径 : {labeled_configs[dataset_name].path}")
        logger.info(f" 📂 无标签路径 : {unlabeled_configs[dataset_name].path}")
        logger.info(f" 🏷️  标签后缀   : {cfg.label_suffix}")

        logger.info("-" * 60)
        logger.info(f" ⚖️  监督损失   : {dual_loss.sup_loss_fn.__class__.__name__}")
        logger.info(f" 👻 伪标签损失 : {dual_loss.pseudo_loss_fn.__class__.__name__}")
        logger.info(f"    ├── 伪标签模式          : {cfg.get('pseudo_label_mode', 'hard')}")
        logger.info(f"    ├── 预热轮数(Ramp-up)   : {cfg.ramp_epochs}")
        logger.info(f"    └── 伪标签最大权重      : {cfg.pseudo_weight}")

        logger.info("-" * 60)
        if cl_cfg and cl_cfg.get("enable", False):
            logger.info(" ⚔️  切面自适应对比学习 (Slice-Adaptive CL) : [🟩 启用]")
            logger.info(f"    ├── 整体损失权重       : {cl_cfg.get('weight', 0.1)}")
            logger.info(f"    ├── InfoNCE 温度系数   : {cl_cfg.get('temperature', 0.1)}")
            logger.info(f"    ├── 预热轮数 (Warmup)  : {cl_cfg.get('warmup_epochs')} 轮 (仅GT更新原型)")
            logger.info(f"    ├── 统一裁剪尺寸       : 2D {cl_cfg.get('patch_size', 16)}x{cl_cfg.get('patch_size', 16)} (GT与伪标签对齐)")
            logger.info(f"    └── 单图采样锚点数     : {cl_cfg.get('num_patches')} 个血管 + {cl_cfg.get('num_patches')} 个背景")
        else:
            logger.info(" ⚔️  切面自适应对比学习 (Slice-Adaptive CL) : [🟥 未启用]")
        logger.info("✨" * 40 + "\n")

    ckpt_cb = ModelCheckpoint(
        dirpath=experiment_dir, monitor="val_DiceMetric", mode="max",
        save_last=True, filename="Epoch{epoch:02d}-{val_DiceMetric:.4f}", save_top_k=1,
        auto_insert_metric_name=False
    )
    ckpt_cb.CHECKPOINT_NAME_LAST = f"{run_name}_last"

    num_devices = len(cfg.trainer.lightning_trainer.get("devices", [1]))
    strategy_opt = DDPStrategy(find_unused_parameters=False) if num_devices > 1 else "auto"
    sync_bn_opt = True if num_devices > 1 else False

    trainer = hydra.utils.instantiate(
        cfg.trainer.lightning_trainer,
        logger=[CleanCSVLogger(save_dir=experiment_dir, name="", version="")],
        callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
        strategy=strategy_opt, sync_batchnorm=sync_bn_opt,
        val_check_interval=cfg.val_frequency, num_sanity_val_steps=0
    )()

    resume_path = cfg.get("resume_ckpt_path", None)
    if resume_path and os.path.exists(resume_path):
        if rank == 0: logger.info(f"🔄 断点重连: {resume_path}")
        if torch.distributed.is_initialized(): torch.distributed.barrier()
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=resume_path)
    else:
        if rank == 0: logger.info("✨ 从头开始全新的双分支半监督训练 (引入对比学习)。")
        trainer.fit(pl_module, train_loader, val_loader)


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()
