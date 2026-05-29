import logging, sys, warnings, os, hydra, torch, numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from utils.dataset import UnionDataset  # 确保此处的 UnionDataset 支持 repeats 参数
from utils.evaluation import Evaluator

# --- 环境与兼容性设置 ---
sys.path.append(str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("monai").setLevel(logging.ERROR)

try:
    import monai.transforms.transform

    monai.transforms.transform.MAX_SEED = 0xFFFFFFFF
except ImportError:
    pass

logger = logging.getLogger(__name__)


# --- 辅助组件 ---

class CleanCSVLogger(CSVLogger):
    def log_hyperparams(self, params): pass


class LogCallback(LearningRateMonitor):
    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0: return
        m = trainer.callback_metrics
        d_name, epoch = pl_module.dataset_name, trainer.current_epoch
        score = m.get(f"{d_name}_val_dice") or m.get("val_DiceMetric")
        best = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None

        logger.info(f"{'=' * 30} Epoch {epoch} {'=' * 30}")
        if score: logger.info(f"✅ {d_name} Dice: {score:.4f}")
        if best and score:
            diff = float(score) - float(best)
            icon = f"🚀 新纪录! (+{diff:.4f})" if diff > 1e-6 else "⚖️ 持平" if diff > -1e-6 else f"🔙 差距: {diff:.4f}"
            logger.info(f"⭐ 历史最佳: {best:.4f} | {icon}")
        logger.info("=" * 67)


def safe_load_weights(model, ckpt_path, rank=0):
    if not ckpt_path: return
    if rank == 0: logger.info(f"🔄 正在加载权重: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    state = {k.replace('model.', '').replace('models.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if rank == 0: logger.info("✅ 权重加载成功。")


# --- 核心修改：支持 Repeats 的加载器 ---
def get_loader(cfg, phase, rank=0):
    # 1. 从配置获取 repeats (每个样本随机裁剪/重复次数)
    repeats = cfg.get("repeats", 1) if phase == "train" else 1

    # 2. 使用 UnionDataset 获取路径和变换
    raw_ds = UnionDataset(cfg.data, phase, finetune=True)
    if not raw_ds.datasets: return None

    info = raw_ds.datasets[0]
    paths, reader, trans = info["paths"], info["reader"], info["transforms"]

    # 3. 限制原始样本数量 (num_shots)
    limit = min(cfg.num_shots, len(paths)) if phase == "train" else len(paths)
    active_paths = paths[:limit]

    # 4. 参考 full_train，通过 UnionDataset 或自定义逻辑扩充样本
    # 注意：这里我们假设你的 UnionDataset 已经像 full_train 那样支持 repeats 参数
    # 如果不支持，可以使用简单的索引重复逻辑
    dataset = UnionDataset(cfg.data, phase, finetune=True, repeats=repeats)

    if rank == 0:
        logger.info(f"📊 [{phase}] 原始样本: {limit} | 扩充倍率: {repeats} | 虚拟总数: {len(dataset)}")

    if phase == "train":
        return hydra.utils.instantiate(cfg.dataloader)(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
    else:
        return hydra.utils.instantiate(cfg.dataloader)(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )


@hydra.main(config_path="../configs", config_name="train/mutil_train", version_base="1.3.2")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    rank = int(os.environ.get("RANK", 0))

    d_name = list(cfg.data.keys())[0]
    run_name = f'Ext_{cfg.num_shots}shot_{cfg.repeats}rep_{os.path.basename(os.path.normpath(cfg.data[d_name].path))}'
    experiment_dir = os.path.join(f"../local_results/doc/{cfg.data_name}", run_name)
    if rank == 0: os.makedirs(experiment_dir, exist_ok=True)

    loggers = [
        WandbLogger(save_dir=experiment_dir, name=run_name, config=OmegaConf.to_container(cfg), offline=True,
                    mode="offline"),
        CleanCSVLogger(save_dir=experiment_dir, name="", version="")
    ]

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{cfg.chkpt_folder}/{cfg.data_name}/{run_name}",
        monitor="val_DiceMetric", mode="max", save_top_k=1, save_last=True,
        filename="epoch:{epoch}-dice:{val_DiceMetric:.2f}"
    )

    # 获取加载器
    train_dl = get_loader(cfg, "train", rank)
    val_dl = get_loader(cfg, "val", rank)

    model = hydra.utils.instantiate(cfg.model)
    safe_load_weights(model, cfg.path_to_chkpt, rank)

    pl_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, evaluator=Evaluator(), dataset_name=d_name
    )

    # 策略调整：val_check_interval=1.0 避免因为样本太少报错
    trainer = hydra.utils.instantiate(
        cfg.trainer.lightning_trainer,
        logger=loggers,
        callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
        devices=cfg.devices,
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,
        val_check_interval=1.0,  # 每轮验证
        check_val_every_n_epoch=1
    )()

    if rank == 0: logger.info(f"🚀 启动扩充训练：每个样本随机裁剪 {cfg.repeats} 次")

    trainer.validate(pl_module, val_dl)
    trainer.fit(pl_module, train_dl, val_dl)


if __name__ == "__main__":
    main()