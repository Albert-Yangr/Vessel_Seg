import logging
import sys
import warnings
import os

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
from pathlib import Path

# 引入 OmegaConf 并在 PyTorch 2.6 中注册安全白名单
from omegaconf import open_dict, DictConfig, ListConfig

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

# 引入项目组件
from utils.evaluation import Evaluator
from utils.mutil_supervision.mutil_dataset import UnionDataset, MutilSupervisionDataset
from 杂项.过去微调.dual2.dual2.model import DualStreamDynUNet
from 杂项.过去微调.dual2.dual2.loss import DualBranchLoss
from 杂项.过去微调.dual2.dual2.dataset import DualStreamDataset, UnlabeledWeakDataset
from 杂项.过去微调.dual2.dual2.module import DualBranchPLModule

sys.path.append(str(Path(__file__).resolve().parent.parent))
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

    # 兼容 PyTorch 2.6 的安全加载
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
    if rank == 0: logger.info(f"✅ 已加载基础权重: {checkpoint_path}")


@hydra.main(config_path="../../../configs", config_name="train/dual_train", version_base="1.3")
def main(cfg):
    seed_everything(cfg.seed, True)
    rank = int(os.environ.get("LOCAL_RANK", 0))

    dataset_name = list(cfg.data.keys())[0]
    base_data_cfg = cfg.data
    run_name = f'{cfg.loss_name}'

    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("🚀 [Dual-Branch CPS] 半监督/弱监督训练任务启动")
        logger.info("=" * 60)

    experiment_dir = f"{cfg.chkpt_folder}/{cfg.data_name}/{os.path.basename(os.path.normpath(cfg.data[dataset_name].path))}/{run_name}"
    if rank == 0: os.makedirs(experiment_dir, exist_ok=True)

    labeled_configs = copy.deepcopy(cfg.data)
    unlabeled_configs = copy.deepcopy(cfg.data)
    val_configs = copy.deepcopy(cfg.data)
    with open_dict(labeled_configs), open_dict(unlabeled_configs), open_dict(val_configs):
        labeled_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, "train")
        unlabeled_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, "train-all")
        val_configs[dataset_name].path = os.path.join(cfg.data[dataset_name].path, )

    train_ds = DualStreamDataset(
        MutilSupervisionDataset(labeled_configs, mode="train", repeats=cfg.repeats, label_suffix=cfg.label_suffix),
        UnlabeledWeakDataset(unlabeled_configs, mode="train", repeats=cfg.repeats)
    )
    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_ds, shuffle=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader, dataset=UnionDataset(val_configs, mode="test", finetune=True),
                                         batch_size=1)

    model = DualStreamDynUNet(hydra.utils.instantiate(cfg.model))
    safe_load_weights(model, cfg.path_to_chkpt, rank)

    target_loss = cfg.loss_configs.slice_loss if cfg.label_suffix == ".slice.nii.gz" else cfg.loss_configs.label_loss

    # =========================================================================
    # 🌟 接入对比学习配置 (从 YAML 提取 contrastive 字段)
    # =========================================================================
    cl_cfg = cfg.loss_configs.get("contrastive", None)

    dual_loss = DualBranchLoss(
        hydra.utils.instantiate(target_loss),
        hydra.utils.instantiate(cfg.loss_configs.pseudo_loss),
        ramp_epochs=cfg.ramp_epochs,
        max_pseudo_weight=cfg.pseudo_weight,
        cl_cfg=cl_cfg  # 将对比学习配置传给 Loss
    )

    pl_module = DualBranchPLModule(model, dual_loss, Evaluator(), cfg.data_name, cfg.optimizer,
                                   loss_configs=cfg.loss_configs)

    # =========================================================================
    # 🌟 增强版：训练启动前，打印数据集、损失函数和【对比学习】的详细配置
    # =========================================================================
    if rank == 0:
        logger.info("\n" + "✨" * 40)
        logger.info(" 📋 [训练核心配置概览]")
        logger.info("✨" * 40)
        logger.info(f" 📁 数据集名称 : {cfg.data_name} ({dataset_name})")
        logger.info(f" 📂 有标签路径 : {labeled_configs[dataset_name].path}")
        logger.info(f" 📂 无标签路径 : {unlabeled_configs[dataset_name].path}")
        logger.info(f" 🏷️  标签后缀   : {cfg.label_suffix}")

        logger.info("-" * 60)
        logger.info(f" ⚖️  监督损失   : {dual_loss.sup_loss_fn.__class__.__name__}")
        if hasattr(dual_loss.sup_loss_fn, 'affinity_weight'):
            logger.info(f"    ├── 亲和力权重(Affinity) : {dual_loss.sup_loss_fn.affinity_weight}")
            logger.info(f"    └── 平滑降噪权重(TV)     : {dual_loss.sup_loss_fn.tv_weight}")

        logger.info(f" 👻 伪标签损失 : {dual_loss.pseudo_loss_fn.__class__.__name__}")
        logger.info(f"    ├── 预热轮数(Ramp-up)    : {cfg.ramp_epochs}")
        logger.info(f"    └── 伪标签最大权重       : {cfg.pseudo_weight}")

        logger.info("-" * 60)
        # 🌟 动态解析并打印双层对比学习组件状态
        if cl_cfg and cl_cfg.get("enable", False):
            logger.info(" ⚔️  双层对比学习 (Bi-Level Contrastive) : [🟩 启用]")
            logger.info(f"    ├── 整体损失权重       : {cl_cfg.get('weight', 0.1)}")
            logger.info(f"    ├── 预热轮数 (Warmup)  : {cl_cfg.get('warmup_epochs', 20)} 轮")
            logger.info(f"    ├── InfoNCE 温度系数   : {cl_cfg.get('temperature', 0.1)}")
            logger.info(f"    ├── 宏观: Patch 尺寸   : {cl_cfg.get('patch_size', '未设置')}")
            logger.info(f"    ├── 宏观: 框采样数量   : {cl_cfg.get('num_patches_per_class', 8)} 个/类")
            logger.info(f"    ├── 微观: 负样本总数   : {cl_cfg.get('pixels_per_patch', 32)} 个/框")
            logger.info(f"    └── 微观: 难例占比     : {cl_cfg.get('hard_neg_ratio', 0.5)} (边缘挖掘率)")
        else:
            logger.info(" ⚔️  双层对比学习 (Bi-Level Contrastive) : [🟥 未启用]")

        logger.info("✨" * 40 + "\n")
    # =========================================================================

    ckpt_cb = ModelCheckpoint(
        dirpath=experiment_dir,
        monitor="val_DiceMetric",
        mode="max",
        save_last=True,
        filename="Epoch{epoch:02d}-{val_DiceMetric:.4f}",
        save_top_k=1,
        auto_insert_metric_name=False
    )
    ckpt_cb.CHECKPOINT_NAME_LAST = f"{run_name}_last"

    # =========================================================================
    # 🌟 核心修复：自动判断单卡还是多卡，防止单卡硬拉 DDP 导致底层死锁闪退
    # =========================================================================
    num_devices = len(cfg.trainer.lightning_trainer.get("devices", [1]))
    strategy_opt = DDPStrategy(find_unused_parameters=False) if num_devices > 1 else "auto"
    sync_bn_opt = True if num_devices > 1 else False

    trainer = hydra.utils.instantiate(
        cfg.trainer.lightning_trainer,
        logger=[CleanCSVLogger(save_dir=experiment_dir, name="", version="")],
        callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
        strategy=strategy_opt,  # 动态判断，单卡设为 auto
        sync_batchnorm=sync_bn_opt,  # 单卡强制关闭 sync_batchnorm
        val_check_interval=cfg.val_frequency,
        num_sanity_val_steps=0
    )()

    # =========================================================================
    # 🌟 真正的断点重连：完美保留优化器与轮数，只切除 Batch 级死锁
    # =========================================================================
    resume_path = cfg.get("resume_ckpt_path", None)

    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            logger.info("=" * 60)
            logger.info(f"🔄 启动原生断点重连: {resume_path}")
            logger.info("=" * 60)

        # 确保多卡同步
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # 直接让 Lightning 接管断点恢复，它会自动处理好一切状态映射
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=resume_path)
    else:
        if rank == 0:
            logger.info("✨ 未检测到断点，从头开始训练。")
        trainer.fit(pl_module, train_loader, val_loader)


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()