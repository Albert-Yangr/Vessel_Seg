import logging
import sys
import warnings
import hydra
import torch
import torch.utils
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, Subset

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from utils.dataset import UnionDataset
from utils.evaluation import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="finetune", version_base="1.3.2")
def main(cfg):
    """
    模型的微调主函数
    - 零样本(zero-shot): num_shots=0，直接测试预训练模型
    - 单样本(one-shot): num_shots=1，使用1个样本微调
    - 少样本(few-shot): num_shots=3，使用3个样本微调
    """

    # 设置随机种子确保实验可复现性
    seed_everything(cfg.seed, True)
    # 设置矩阵乘法精度平衡速度和精度
    torch.set_float32_matmul_precision("medium")

    # 构建运行名称，包含关键实验信息
    dataset_name = list(cfg.data.keys())[0]  # 获取数据集名称
    run_name = f'finetune_{cfg.num_shots}shot_{dataset_name}_' + cfg.run_name

    # 初始化Weights & Biases日志记录器
    wnb_logger = WandbLogger(
        project=cfg.wandb_project,  # 项目名称
        name=run_name,  # 运行名称
        config=OmegaConf.to_container(cfg),  # 记录完整配置
        offline=cfg.offline,  # 离线模式开关
    )

    # 设置训练回调函数
    lr_monitor = LearningRateMonitor()  # 学习率监控
    monitor_metric = "val_DiceMetric"  # 监控指标（Dice系数）

    # 模型检查点回调 - 保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.chkpt_folder + "/" + cfg.wandb_project + "/" + run_name,  # 保存路径
        monitor=monitor_metric,  # 监控的指标
        save_top_k=1,  # 只保存最好的1个模型
        mode="max",  # 指标越大越好
        filename=f"{run_name}_" + "{step}_{" + monitor_metric + ":.2f}",  # 文件名格式
        auto_insert_metric_name=True,  # 自动插入指标名
        save_last=True  # 同时保存最后一个epoch的模型
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"

    # 初始化PyTorch Lightning训练器
    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": wnb_logger,  # 日志记录器
        "callbacks": [lr_monitor, checkpoint_callback],  # 回调函数
        "devices": cfg.devices  # 训练设备
    }
    trainer = trainer(**trainer_additional_kwargs)

    # 初始化数据加载器 - 关键部分对应论文中的实验设置
    # 训练数据集：使用UnionDataset并限制样本数量
    train_dataset = UnionDataset(cfg.data, "fine-tuning", finetune=True)
    train_dataset = Subset(train_dataset, range(cfg.num_shots))  # 限制样本数量

    # 使用随机采样器并进行重复采样，模拟论文中的少样本设置
    random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=int(1200))
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset, sampler=random_sampler)
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")

    # 验证数据集
    val_dataset = UnionDataset(cfg.data, "val", finetune=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1)
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # 测试数据集
    test_dataset = UnionDataset(cfg.data, "test", finetune=True)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=test_dataset, batch_size=1)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # 初始化模型 - 对应论文中的vesselFM架构
    model = hydra.utils.instantiate(cfg.model)
    # 加载预训练权重（如果指定了检查点路径）
    if cfg.path_to_chkpt is not None:
        chkpt = torch.load(cfg.path_to_chkpt, map_location=f'cuda:{cfg.devices[0]}')
        # 处理状态字典键名，移除"models."前缀
        model_chkpt = chkpt  # 直接使用整个检查点
        model.load_state_dict(model_chkpt)
        logger.info(f"Loaded pretrained weights from {cfg.path_to_chkpt}")

    # 初始化Lightning模块 - 封装训练逻辑
    evaluator = Evaluator()  # 评估器，用于计算Dice和clDice指标
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model,
        evaluator=evaluator,
        dataset_name=dataset_name
    )
    # 开始训练和评估流程
    wnb_logger.watch(model, log="all", log_freq=20)  # 监控模型参数

    # 根据样本数量选择不同的实验模式
    if cfg.num_shots == 0:
        # 零样本评估：直接测试预训练模型
        logger.info("Starting zero-shot evaluation")
        trainer.test(lightning_module, test_loader)
    else:
        # 少样本微调：验证→训练→测试完整流程
        logger.info("Starting training")
        trainer.validate(lightning_module, val_loader)  # 初始验证
        trainer.fit(lightning_module, train_loader, val_loader)  # 训练
        logger.info("Finished training")
        # 使用最佳检查点进行最终测试
        trainer.test(lightning_module, test_loader, ckpt_path="best")


if __name__ == "__main__":
    # 设置标准输出缓冲，确保日志实时显示
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()