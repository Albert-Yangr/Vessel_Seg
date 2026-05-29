'''

该文件使用utils.dataset和utils.LPMoE以及mutil_train配置的情况下为原始版本

使用utils.dataset和utils.lpmoe2以及lpmoe_train配置的情况下为血管分类后的版本

使用utils.lpmoe3.dataset和utils.lpmoe3以及lpmoe3_train配置的情况下为原始版本

'''


import logging
import sys
import warnings
import os
import numpy as np
from pathlib import Path

# 获取当前脚本的绝对路径
current_file_path = Path(__file__).resolve()
# 获取项目根目录 (即 train 文件夹的上一级)
project_root = current_file_path.parent.parent
# 将项目根目录添加到 python 搜索路径中
sys.path.append(str(project_root))

# ==========================================
# 【关键修复】MONAI 与 NumPy 版本兼容性修复
try:
    import monai.transforms.transform

    # 强制修改 MONAI 内部的 MAX_SEED，防止 NumPy 报错 (OverflowError)
    monai.transforms.transform.MAX_SEED = 0xFFFFFFFF
except ImportError:
    pass
# ==========================================

import hydra
import torch
import torch.utils
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from utils.dataset import UnionDataset
from utils.evaluation import Evaluator
# 【新增】导入 LPMoE 模型定义
from utils.LPMoE.lpmoe_unet import LPMoE_VesselNet

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def smart_load_and_freeze(model, ckpt_path, lpmoe_keyword="experts"):
    """
    【智能权重加载与冻结策略 - 终极修复版】
    Args:
        model: LPMoE 模型实例
        ckpt_path: 权重路径
        lpmoe_keyword: 用于识别分支模块的关键词 (默认 'experts'，也包含 'gate', 'adapter')
    Returns:
        model: 处理后的模型
        run_prefix: 文件夹前缀 ("Frozen_" 或 "Global_")
    """
    if not ckpt_path or ckpt_path == "None":
        logger.info("⚠️ 无权重路径，从头训练 (Scratch Training)")
        return model, ""

    logger.info(f"📂 [Smart Load] 正在分析权重文件: {ckpt_path}")

    # 1. 加载权重
    try:
        chkpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.error(f"❌ 权重加载失败: {e}")
        raise e

    # 提取 state_dict
    if isinstance(chkpt, dict) and 'state_dict' in chkpt:
        state_dict = chkpt['state_dict']
        is_lightning = True
    elif isinstance(chkpt, dict) and 'model_state_dict' in chkpt:
        state_dict = chkpt['model_state_dict']
        is_lightning = False
    else:
        state_dict = chkpt
        is_lightning = False

    # 清洗前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if is_lightning and k.startswith('model.'):
            new_k = k.replace('model.', '', 1)
        elif k.startswith('models.'):
            new_k = k.replace('models.', '', 1)
        clean_state_dict[new_k] = v

    # 2. 核心判断：检查是否有 LPMoE 特有层
    has_lpmoe_modules = any((lpmoe_keyword in k or 'gate' in k or 'adapter' in k) for k in clean_state_dict.keys())

    run_prefix = ""

    if not has_lpmoe_modules:
        # === CASE 1: 仅有基础权重 (Frozen Mode) ===
        # 直接调用模型自带的 load_backbone_weights 方法
        logger.info(f"❄️ 检测到【基础权重】(未发现 '{lpmoe_keyword}' 等分支模块)。")
        logger.info("   -> 策略: 调用 load_backbone_weights 加载主干，并冻结。")

        try:
            model.load_backbone_weights(clean_state_dict)
            logger.info("   -> ✅ 主干权重加载成功 (使用 load_backbone_weights)")
        except Exception as e:
            logger.error(f"❌ 主干加载失败: {e}")
            raise e

        # 调用模型自带的冻结方法
        model.freeze_backbone()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"   -> ❄️ 主干已冻结，当前可训练参数量: {trainable_params}")

        run_prefix = "Frozen_"

    else:
        # === CASE 2: 完整权重 (Global Mode) ===
        logger.info(f"🔥 检测到【完整权重】(包含 '{lpmoe_keyword}' 等分支模块)。")
        logger.info("   -> 策略: 全局微调 (Global Fine-tuning)，所有层参与训练。")

        # 对于完整权重，直接 load_state_dict 即可
        model.load_state_dict(clean_state_dict, strict=False)

        # 强制解冻所有层
        for param in model.parameters():
            param.requires_grad = True

        logger.info("   -> ✅ 所有层已解冻，准备进行全量微调。")
        run_prefix = "Global_"

    return model, run_prefix


def _log_validation_details(phase, trainer, pl_module, dataset_name):
    """记录验证详细结果"""
    if trainer.global_rank != 0: return

    current_metrics = trainer.callback_metrics
    dice_score = current_metrics.get(f"{dataset_name}_val_dice", None)
    val_dice_metric = current_metrics.get("val_DiceMetric", None)
    val_loss = current_metrics.get(f"{dataset_name}_val_loss", None)

    logger.info("📈 " + "=" * 50)
    logger.info(f"📋 {phase}结果摘要")
    logger.info("📈 " + "=" * 50)
    logger.info(f"📊 数据集: {dataset_name}")
    if dice_score is not None:
        logger.info(f"🎯 数据集Dice: {dice_score:.4f}")
    if val_dice_metric is not None:
        logger.info(f"⭐ 综合Dice指标: {val_dice_metric:.4f}")
    if val_loss is not None:
        logger.info(f"📉 验证损失: {val_loss:.4f}")
    logger.info("📈 " + "=" * 50)


def _log_test_summary(trainer, pl_module, dataset_name):
    """记录测试结果摘要"""
    if trainer.global_rank != 0: return
    current_metrics = trainer.callback_metrics
    test_dice = current_metrics.get(f"{dataset_name}_test_dice", None)
    test_dice_metric = current_metrics.get("test_DiceMetric", None)
    logger.info("🎉 " + "=" * 60)
    logger.info("🏆 最终测试结果报告")
    if test_dice is not None: logger.info(f"✅ 测试集Dice分数: {test_dice:.4f}")
    if test_dice_metric is not None: logger.info(f"🏅 最终Dice指标: {test_dice_metric:.4f}")
    logger.info("🎉 " + "=" * 60)


@hydra.main(config_path="../../configs", config_name="train/mutil_train", version_base="1.3.2")
def main(cfg):
    """
    Controllable-LPMoE 微调主函数 (V3.7 Dynamic Batch Size)
    【已修复】强制随机种子与确定性训练
    """

    # ==========================================
    # 1. 【强制修复】随机种子与确定性设置
    # ==========================================
    # 优先使用配置文件的 seed，如果没有则默认 42
    FIXED_SEED = 42
    if hasattr(cfg, 'seed') and cfg.seed is not None:
        FIXED_SEED = int(cfg.seed)

    # 设置所有库的种子 (workers=True 确保 DataLoader 子进程也一致)
    seed_everything(FIXED_SEED, workers=True)

    # 强制 PyTorch 使用确定性算法 (会牺牲少量性能，但保证可复现)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 注意: medium 精度在不同硬件上可能有微小差异，为了极致复现可注释掉，
    # 但为了保持之前的训练速度，这里保留，主要依靠 deterministic=True 来约束
    torch.set_float32_matmul_precision("medium")

    global_rank = int(os.environ.get("RANK", 0))
    if global_rank == 0:
        logger.info("=" * 40)
        logger.info(f"🔒 [Reproducibility] 随机种子已锁定为: {FIXED_SEED}")
        logger.info(f"🔒 [Reproducibility] Deterministic 模式: 开启")
        logger.info(f"🔒 [Reproducibility] CUDNN Benchmark: 关闭")
        logger.info("=" * 40)

    # 构建运行名称
    dataset_name = list(cfg.data.keys())[0]
    full_data_path = cfg.data[dataset_name].path
    last_folder_name = os.path.basename(os.path.normpath(full_data_path))
    base_run_name = f'{cfg.loss_name}_{cfg.num_shots}shot_{last_folder_name}'

    cfg.offline = True
    save_root_dir = f"{project_root}/local_results/logs/{cfg.data_name}"
    if global_rank == 0:
        os.makedirs(save_root_dir, exist_ok=True)
        logger.info(f"📂 日志存储路径已设置为: {save_root_dir}")

    # 初始化日志记录器
    wnb_logger = WandbLogger(
        save_dir=save_root_dir,
        project=cfg.wandb_project,
        name=base_run_name,
        config=OmegaConf.to_container(cfg),
        offline=True,
        mode="offline"
    )

    csv_logger = CSVLogger(
        save_dir=save_root_dir,
        name=base_run_name,
        version="version_0"
    )

    # ---------------------------------------------------------
    # 动态计算步数策略
    # ---------------------------------------------------------
    target_devices = cfg.devices
    num_devices = len(target_devices)
    base_total_steps = cfg.trainer.lightning_trainer.max_steps
    actual_max_steps = int(base_total_steps // num_devices)

    # 评估间隔调整为 总步数 // 25
    actual_val_interval = max(1, int(actual_max_steps // 25))

    if global_rank == 0:
        logger.info("=" * 40)
        logger.info(f"🧮 动态训练策略调整 (GPU数量: {num_devices})")
        logger.info(f"   - YAML基准步数: {base_total_steps}")
        logger.info(f"   - 实际训练步数: {actual_max_steps}")
        logger.info(f"   - 评估间隔: {actual_val_interval} steps (1/25)")
        logger.info("=" * 40)

    # =========================================================
    # 【Auto-Tuning 1】 初始默认高学习率 (用于 Frozen 模式)
    # =========================================================
    TARGET_LR = 1e-3

    with open_dict(cfg):
        # 1. 设置默认 Optimizer Learning Rate
        if hasattr(cfg.trainer.lightning_module, 'optimizer_factory'):
            opt_conf = cfg.trainer.lightning_module.optimizer_factory
            old_lr = opt_conf.get('lr', 'Default')
            opt_conf.lr = TARGET_LR
            if global_rank == 0:
                logger.info(f"🚀 [Init LR] 学习率初始化为: {old_lr} -> {TARGET_LR} (Frozen模式预设)")

        # 2. 修正 Scheduler T_max
        if hasattr(cfg.trainer.lightning_module, 'scheduler_configs'):
            sched_configs = cfg.trainer.lightning_module.scheduler_configs
            if 'cosine_annealing_few' in sched_configs:
                sched = sched_configs.cosine_annealing_few.scheduler
                sched.T_max = actual_max_steps

    # 设置回调函数
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_DiceMetric"

    class ValidationResultCallback(LearningRateMonitor):
        def on_validation_end(self, trainer, pl_module):
            if trainer.global_rank != 0: return

            current_metrics = trainer.callback_metrics
            dice_score = current_metrics.get(f"{dataset_name}_val_dice", None)
            val_dice_metric = current_metrics.get("val_DiceMetric", None)
            val_loss = current_metrics.get(f"{dataset_name}_val_loss", None)
            current_epoch = trainer.current_epoch

            logger.info("=" * 60)
            logger.info(f"📊 验证结果报告 (Epoch {current_epoch})")
            if dice_score is not None: logger.info(f"✅ {dataset_name} Dice: {dice_score:.4f}")
            if val_dice_metric is not None: logger.info(f"🏆 验证Dice指标: {val_dice_metric:.4f}")
            if val_loss is not None: logger.info(f"📉 验证损失值: {val_loss:.4f}")

            if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback is not None:
                best_dice = trainer.checkpoint_callback.best_model_score
                if best_dice is not None:
                    logger.info(f"⭐ 历史最佳Dice: {best_dice:.4f}")
                    if dice_score is not None:
                        current_val = float(dice_score)
                        best_val = float(best_dice)
                        diff = current_val - best_val
                        if diff > 0:
                            logger.info(f"🚀 新纪录! 提升: +{diff:.4f}")
                        elif diff == 0:
                            logger.info(f"⚖️  持平历史最佳")
                        else:
                            logger.info(f"🔙 距历史最佳: {diff:.4f}")
            logger.info("=" * 60)

    # 初始 Checkpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.chkpt_folder}/{cfg.data_name}/{last_folder_name}/{base_run_name}",
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        filename="{step}_{" + monitor_metric + ":.2f}_" + f"{num_devices}GPUs",
        auto_insert_metric_name=True,
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last"

    validation_callback = ValidationResultCallback()

    # ---------------------------------------------------------
    # 2. 【强制修复】Trainer 确定性配置
    # ---------------------------------------------------------
    trainer_cls = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer = trainer_cls(
        logger=[wnb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback, validation_callback],
        max_steps=actual_max_steps,
        val_check_interval=actual_val_interval,
        devices=target_devices,
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=False,
        deterministic=True  # <--- 关键修改：强制使用确定性算法
    )

    # ---------------------------------------------------------
    # 数据集加载
    # ---------------------------------------------------------
    class FewShotInMemoryDataset(torch.utils.data.Dataset):
        def __init__(self, data_list, transform):
            self.data = data_list
            self.transform = transform

        def __len__(self): return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            transformed = self.transform(item)
            return transformed['Image'], transformed['Mask'] > 0

    # Train (准备数据)
    raw_train_dataset = UnionDataset(cfg.data, "train", finetune=True)
    d_info = raw_train_dataset.datasets[0]
    data_paths = d_info["paths"]
    reader = d_info["reader"]
    data_transform = d_info["transforms"]

    subset_data_list = []
    shots_to_load = min(cfg.num_shots, len(data_paths))
    if global_rank == 0: logger.info(f"🚀 加载 {shots_to_load} 个训练样本到内存...")

    for i in range(shots_to_load):
        sample_path = data_paths[i]
        img_path = [p for p in sample_path.iterdir() if 'img' in p.name][0]
        mask_path = [p for p in sample_path.iterdir() if 'label' in p.name][0]
        img = reader.read_images(str(img_path))[0].astype(np.float32)
        mask = reader.read_images(str(mask_path))[0].astype(bool)
        subset_data_list.append({'Image': img, 'Mask': mask})

    train_dataset = FewShotInMemoryDataset(data_list=subset_data_list, transform=data_transform)

    # ---------------------------------------------------------
    # 3. 【强制修复】Sampler 绑定 Generator
    # ---------------------------------------------------------
    # 创建独立的 Generator，确保采样序列一致
    generator = torch.Generator()
    generator.manual_seed(FIXED_SEED)

    # 采样器配置
    total_samples_per_epoch = int(1e5)
    samples_per_gpu = total_samples_per_epoch // num_devices

    random_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=samples_per_gpu,
        generator=generator  # <--- 关键修改：显式传入 Generator
    )

    # Val
    def _load_split_to_memory(cfg, phase, global_rank):
        raw_dataset = UnionDataset(cfg.data, phase, finetune=True)
        if not raw_dataset.datasets or len(raw_dataset) == 0: return None
        d_info = raw_dataset.datasets[0]
        d_list = []
        if global_rank == 0: logger.info(f"🚀 加载 {phase} 集...")
        for s_path in d_info["paths"]:
            i_path = [p for p in s_path.iterdir() if 'img' in p.name][0]
            m_path = [p for p in s_path.iterdir() if 'label' in p.name][0]
            img = d_info["reader"].read_images(str(i_path))[0].astype(np.float32)
            mask = d_info["reader"].read_images(str(m_path))[0].astype(bool)
            d_list.append({'Image': img, 'Mask': mask})
        return FewShotInMemoryDataset(data_list=d_list, transform=d_info["transforms"])

    val_dataset = _load_split_to_memory(cfg, "val", global_rank)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1) if val_dataset else None

    # Test
    test_dataset = UnionDataset(cfg.data, "test", finetune=True)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1)

    # =========================================================
    # 【模型构建与智能加载】
    # =========================================================
    base_model_config_path = f"{project_root}/configs/model/dyn_unet_base.yaml"
    if global_rank == 0: logger.info(f"📄 读取基础配置: {base_model_config_path}")
    base_model_cfg = OmegaConf.load(base_model_config_path)

    if global_rank == 0: logger.info("🏗️ 正在构建 LPMoE_VesselNet...")
    model = LPMoE_VesselNet(base_config=base_model_cfg, num_classes=1)

    # --- 调用智能加载 ---
    # 【核心修改】初始化默认 Batch Size (针对 Scratch 训练)
    train_batch_size = 4

    if cfg.path_to_chkpt is not None:
        model, path_prefix = smart_load_and_freeze(model, cfg.path_to_chkpt, lpmoe_keyword="experts")

        if path_prefix:
            # 1. 更新保存路径
            final_run_name = f"{path_prefix}{base_run_name}"
            new_save_path = f"{cfg.chkpt_folder}/{cfg.data_name}/{last_folder_name}/{final_run_name}"
            checkpoint_callback.dirpath = new_save_path
            checkpoint_callback.CHECKPOINT_NAME_LAST = f"{final_run_name}_last"

            if global_rank == 0:
                logger.info(f"📂 [Path Update] 保存路径更新为: {new_save_path}")

            # 2. 如果是 Global 模式，降低学习率到 1e-5，BatchSize=2
            if path_prefix == "Global_":
                train_batch_size = 2
                GLOBAL_LR = 1e-5
                with open_dict(cfg):
                    if hasattr(cfg.trainer.lightning_module, 'optimizer_factory'):
                        cfg.trainer.lightning_module.optimizer_factory.lr = GLOBAL_LR
                        if global_rank == 0:
                            logger.info(f"📉 [Adaptive LR] 检测到 Global 模式，学习率下调至: {GLOBAL_LR}")

            # 3. 如果是 Frozen 模式，BatchSize=4，学习率保持 1e-3
            elif path_prefix == "Frozen_":
                train_batch_size = 4
                if global_rank == 0:
                    logger.info(f"🚀 [Adaptive LR] 保持 Frozen 模式高学习率: {TARGET_LR}")

            if global_rank == 0:
                logger.info(
                    f"⚖️ [Dynamic Batch Size] 检测到 {path_prefix[:-1]} 模式，设置训练 Batch Size = {train_batch_size}")


    else:
        if global_rank == 0:
            logger.info(f"⚠️ [Dynamic Batch Size] 暂无权重 (Scratch)，使用默认 Batch Size = {train_batch_size}")

    # 【核心修改】实例化 Train Loader (在确定 batch_size 后)
    train_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=train_dataset,
        sampler=random_sampler,
        batch_size=train_batch_size)

    # 初始化 LightningModule
    evaluator = Evaluator()
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model,
        evaluator=evaluator,
        dataset_name=dataset_name
    )

    # 训练流程
    if not cfg.offline:
        if global_rank == 0: wnb_logger.watch(model, log="all", log_freq=20)
    else:
        if global_rank == 0: logger.info("离线模式：跳过模型参数监控")

    if cfg.num_shots == 0:
        if global_rank == 0: logger.info("Starting zero-shot evaluation")
        trainer.test(lightning_module, test_loader)
    else:
        if global_rank == 0:
            logger.info("Starting training")
            logger.info("🔍 进行初始验证...")

        trainer.validate(lightning_module, val_loader)
        if global_rank == 0:
            _log_validation_details("初始验证", trainer, lightning_module, dataset_name)
            logger.info("🚀 开始模型训练...")

        trainer.fit(lightning_module, train_loader, val_loader)

        if global_rank == 0:
            logger.info("Finished training")
            logger.info("🧪 进行最终测试...")

        trainer.test(lightning_module, test_loader, ckpt_path="best")
        if global_rank == 0:
            _log_test_summary(trainer, lightning_module, dataset_name)
            logger.info(f"实验完成！日志保存在：{save_root_dir}")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()