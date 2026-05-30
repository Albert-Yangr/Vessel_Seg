import copy
import logging
import os
import sys
import warnings
from pathlib import Path

# =============================================================================
# 基础双分支 CPS 训练入口
# -----------------------------------------------------------------------------
# 这个脚本对应 configs/train/dual_train.yaml。
#
# 主要职责：
#   1. 读取 Hydra 配置。
#   2. 构建有标签数据流、无标签数据流和验证数据流。
#   3. 构建双分支 DynUNet。
#   4. 构建监督损失 + 伪标签损失。
#   5. 封装成 LightningModule，并交给 Lightning Trainer 训练。
#   6. 保存 checkpoint、train_config.yaml 和实验记录。
# =============================================================================

sys.path.append(str(Path(__file__).resolve().parent.parent))

# 限制底层数值库线程数，避免 DataLoader 多进程时 CPU 线程爆炸。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from utils.dual_branch.dataset import (
    DualStreamDataset,
    MutilSupervisionDataset,
    UnionDataset,
    UnlabeledWeakDataset,
)
from utils.dual_branch.loss import DualBranchLoss
from utils.dual_branch.model import DualStreamDynUNet
from utils.dual_branch.module import DualBranchPLModule
from utils.evaluation import Evaluator
from utils.experiment_tracker import save_experiment_record

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class CleanCSVLogger(CSVLogger):
    """关闭 CSVLogger 的 hparams 写入，只保留 metrics.csv。"""

    def log_hyperparams(self, params):
        pass


class LogCallback(LearningRateMonitor):
    """验证结束后在控制台打印当前 Dice 和历史 best Dice。"""

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        score = trainer.callback_metrics.get("val_DiceMetric")
        best = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None

        logger.info("=" * 60)
        logger.info(f"Epoch {trainer.current_epoch} | Step {trainer.global_step}")
        if score is not None:
            logger.info(f"Current val Dice: {float(score):.4f}")
        if best is not None:
            logger.info(f"Best val Dice: {float(best):.4f}")
        logger.info("=" * 60)


def safe_load_weights(model, checkpoint_path, rank=0):
    """
    可选加载预训练权重。

    这里主要用于把普通 nnU-Net / DynUNet 权重加载到 DualStreamDynUNet 的
    base_model 中。辅助 decoder 会保持当前初始化。
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return

    if rank == 0:
        logger.info(f"Loading base weights: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    new_state = {
        f"base_model.{k.replace('model.', '').replace('net.', '')}": v
        for k, v in state.items()
        if f"base_model.{k.replace('model.', '').replace('net.', '')}" in model_state
    }
    model.load_state_dict(new_state, strict=False)

    if rank == 0:
        logger.info(f"Loaded {len(new_state)} tensors into base_model.")


def _num_devices(devices):
    """兼容 devices=[1]、devices=1 等 Lightning 配置写法。"""
    if isinstance(devices, (list, tuple, ListConfig)):
        return len(devices)
    return int(devices)


@hydra.main(config_path="../configs", config_name="train/dual_train", version_base="1.3")
def main(cfg):
    seed_everything(cfg.seed, True)
    rank = int(os.environ.get("LOCAL_RANK", 0))

    dataset_name = list(cfg.data.keys())[0]
    run_name = str(cfg.loss_name)

    # checkpoint 目录保留数据集名、原始数据文件夹名和实验名，便于回溯。
    data_folder_name = os.path.basename(os.path.normpath(cfg.data[dataset_name].path))
    experiment_dir = f"{cfg.chkpt_folder}/{cfg.data_name}/{data_folder_name}/{run_name}"

    if rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)
        config_save_path = os.path.join(experiment_dir, "train_config.yaml")
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=False))
        save_experiment_record(cfg, __file__, experiment_dir, logger)

        logger.info("=" * 60)
        logger.info("Dual-Branch CPS training starts")
        logger.info(f"Config saved to: {config_save_path}")
        logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 构建数据路径
    # -------------------------------------------------------------------------
    labeled_configs = copy.deepcopy(cfg.data)
    unlabeled_configs = copy.deepcopy(cfg.data)
    val_configs = copy.deepcopy(cfg.data)

    with open_dict(labeled_configs), open_dict(unlabeled_configs), open_dict(val_configs):
        labeled_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, "train")
        unlabeled_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, "train-all")
        val_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path)

    # -------------------------------------------------------------------------
    # 2. 构建双流训练集
    # -------------------------------------------------------------------------
    labeled_ds = MutilSupervisionDataset(
        labeled_configs,
        mode="train",
        repeats=cfg.repeats,
        label_suffix=cfg.label_suffix,
    )
    unlabeled_ds = UnlabeledWeakDataset(
        unlabeled_configs,
        mode="train",
        repeats=cfg.repeats,
    )
    train_ds = DualStreamDataset(labeled_ds, unlabeled_ds)

    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_ds, shuffle=True)

    # 验证集使用完整 label，batch_size 固定为 1，配合 sliding-window inference。
    val_loader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=UnionDataset(val_configs, mode="test", finetune=True),
        batch_size=1,
    )

    # -------------------------------------------------------------------------
    # 3. 构建模型
    # -------------------------------------------------------------------------
    base_model = hydra.utils.instantiate(cfg.model)
    model = DualStreamDynUNet(base_model)
    safe_load_weights(model, cfg.path_to_chkpt, rank)

    # -------------------------------------------------------------------------
    # 4. 构建损失
    # -------------------------------------------------------------------------
    target_loss = cfg.loss_configs.slice_loss if ".slice" in cfg.label_suffix else cfg.loss_configs.label_loss
    dual_loss = DualBranchLoss(
        hydra.utils.instantiate(target_loss),
        hydra.utils.instantiate(cfg.loss_configs.pseudo_loss),
        ramp_epochs=cfg.ramp_epochs,
        max_pseudo_weight=cfg.pseudo_weight,
        pseudo_label_mode=cfg.get("pseudo_label_mode", "hard"),
    )

    pl_module = DualBranchPLModule(model, dual_loss, Evaluator(), cfg.data_name, cfg.optimizer)

    if rank == 0:
        logger.info("Training setup")
        logger.info(f"Dataset: {cfg.data_name} ({dataset_name})")
        logger.info(f"Labeled path: {labeled_configs[dataset_name].path}")
        logger.info(f"Unlabeled path: {unlabeled_configs[dataset_name].path}")
        logger.info(f"Label suffix: {cfg.label_suffix}")
        logger.info(f"Supervised loss: {dual_loss.sup_loss_fn.__class__.__name__}")
        logger.info(f"Pseudo loss: {dual_loss.pseudo_loss_fn.__class__.__name__}")
        logger.info(f"Pseudo label mode: {cfg.get('pseudo_label_mode', 'hard')}")
        logger.info(f"Ramp epochs: {cfg.ramp_epochs}")
        logger.info(f"Pseudo weight: {cfg.pseudo_weight}")

    # -------------------------------------------------------------------------
    # 5. checkpoint 和 Trainer
    # -------------------------------------------------------------------------
    ckpt_cb = ModelCheckpoint(
        dirpath=experiment_dir,
        monitor="val_DiceMetric",
        mode="max",
        save_last=True,
        filename="Epoch{epoch:02d}-{val_DiceMetric:.4f}",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    ckpt_cb.CHECKPOINT_NAME_LAST = f"{run_name}_last"

    devices = cfg.trainer.lightning_trainer.get("devices", [1])
    num_devices = _num_devices(devices)
    strategy_opt = DDPStrategy(find_unused_parameters=False) if num_devices > 1 else "auto"
    sync_bn_opt = num_devices > 1

    trainer = hydra.utils.instantiate(
        cfg.trainer.lightning_trainer,
        logger=[CleanCSVLogger(save_dir=experiment_dir, name="", version="")],
        callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
        strategy=strategy_opt,
        sync_batchnorm=sync_bn_opt,
        val_check_interval=cfg.val_frequency,
        num_sanity_val_steps=0,
    )()

    # -------------------------------------------------------------------------
    # 6. 启动训练，可选从 Lightning checkpoint 恢复
    # -------------------------------------------------------------------------
    resume_path = cfg.get("resume_ckpt_path", None)
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            logger.info(f"Resume from checkpoint: {resume_path}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=resume_path)
    else:
        if rank == 0:
            logger.info("Start training from scratch.")
        trainer.fit(pl_module, train_loader, val_loader)


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()
