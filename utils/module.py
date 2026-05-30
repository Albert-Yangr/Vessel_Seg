import logging
import os

import torch
import lightning
import numpy as np
from monai.inferers.inferer import SlidingWindowInfererAdapt

logger = logging.getLogger(__name__)


class PLModule(lightning.LightningModule):
    """
    实现了标准的训练、验证流程
    支持多GPU训练和分布式日志记录
    """

    def __init__(
            self,
            model: torch.nn.Module,
            loss,
            optimizer_factory,
            prediction_threshold: float,
            scheduler_configs=None,
            evaluator=None
    ):
        """
        初始化基础Lightning模块

        Args:
            model: 分割模型（vesselFM的UNet架构）
            loss: 损失函数（Dice + CrossEntropy组合）
            optimizer_factory: 优化器工厂函数
            prediction_threshold: 预测阈值（用于二值化）
            scheduler_configs: 学习率调度器配置
            evaluator: 评估器（计算Dice/clDice指标）
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs
        self.prediction_threshold = prediction_threshold
        # 获取当前进程的rank，用于分布式训练中的日志控制
        self.rank = 0 if "LOCAL_RANK" not in os.environ else os.environ["LOCAL_RANK"]
        self.evaluator = evaluator

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        - 使用AdamW优化器
        - 学习率10^-4，衰减到10^-5
        - 线性warmup和衰减策略
        """
        optimizer = self.optimizer_factory(params=self.parameters())

        if self.scheduler_configs is not None:
            schedulers = []
            logger.info(f"Initializing schedulers: {self.scheduler_configs}")
            # 遍历所有调度器配置
            for scheduler_name, scheduler_config in self.scheduler_configs.items():
                if scheduler_config is None:
                    continue  # 跳过微调时的空配置

                logger.info(f"Initializing scheduler: {scheduler_name}")
                # 实例化调度器
                scheduler_config["scheduler"] = scheduler_config["scheduler"](optimizer=optimizer)
                scheduler_config = dict(scheduler_config)
                schedulers.append(scheduler_config)
            return [optimizer], schedulers
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        训练步骤 - 前向传播和损失计算
        Args:
            batch: 包含图像和掩码的批次数据
            batch_idx: 批次索引

        Returns:
            loss: 计算得到的损失值
        """
        image, mask = batch
        # 模型前向传播
        pred_mask = self.model(image)
        # 计算损失（Dice + CrossEntropy组合）
        loss = self.loss(pred_mask, mask)
        # 记录训练损失（只在rank 0进程记录日志）
        self.log(f"train_loss", loss.item(), logger=(self.rank == 0))
        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤 - 评估模型在验证集上的性能
        - Dice系数：分割重叠度
        - clDice：拓扑结构保持度
        """
        image, mask, name = batch
        pred_mask = self.model(image)
        loss = self.loss(pred_mask, mask)
        self.log("val_loss", loss.item(), logger=(self.rank == 0))

        # 计算评估指标（Dice和clDice）
        metrics = self.evaluator.estimate_metrics(
            pred_mask.sigmoid().squeeze(),  # 应用sigmoid并去除批次维度
            mask.squeeze(),
            threshold=self.prediction_threshold
        )

        # 记录所有评估指标
        for metric, value in metrics.items():
            value = value.item() if isinstance(value, (torch.Tensor, np.ndarray)) else value
            self.log(f"val_{name[0]}_{metric}", value, logger=(self.rank == 0))


class PLModuleFinetune(PLModule):
    """
    微调专用的Lightning模块
    扩展基础模块，支持少样本微调场景
    - 零样本（num_shots=0）
    - 单样本（num_shots=1）
    - 少样本（num_shots=3）
    """

    def __init__(
            self,
            dataset_name: str = None,
            input_size: tuple = None,
            batch_size: int = None,
            num_shots: int = None,
            *args,
            **kwargs
    ):
        """
        初始化微调模块

        Args:
            dataset_name: 数据集名称（用于日志记录）
            input_size: 输入尺寸（滑动窗口大小）
            batch_size: 批次大小
            num_shots: 少样本数量
        """
        # 移除dataset_name避免传递给父类
        self.dataset_name = dataset_name
        logger.info(f"Dataset name: {self.dataset_name}")
        super().__init__(*args, **kwargs)

        # 初始化滑动窗口推理器，用于处理大尺寸验证/测试数据
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=input_size,  # 滑动窗口大小
            sw_batch_size=batch_size,  # 批大小
            overlap=0.5,  # 50%重叠率
        )
        self.num_shots = num_shots  # 记录少样本数量

    def validation_step(self, batch, batch_idx):
        """
        验证步骤 - 使用滑动窗口推理处理大尺寸数据
        在微调场景下，验证数据可能比训练patch大，需要滑动窗口处理
        """
        image, mask = batch
        with torch.no_grad():  # 禁用梯度计算，节省内存
            # 使用滑动窗口进行推理
            pred_mask = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_mask, mask)
            self.log(f"{self.dataset_name}_val_loss", loss.item())

            # 计算评估指标（使用快速模式）
            metrics = self.evaluator.estimate_metrics(
                pred_mask.sigmoid().squeeze(),
                mask.squeeze(),
                threshold=self.prediction_threshold,
                fast=True  # 快速模式，适用于验证阶段
            )

            # 记录数据集特定的指标
            for name, value in metrics.items():
                value = value.item() if isinstance(value, (torch.Tensor, np.ndarray)) else value
                self.log(f"{self.dataset_name}_val_{name}", value)

                # 特别记录Dice指标用于模型选择
                if name == "dice":
                    self.log("val_DiceMetric", value)  # 用于ModelCheckpoint回调

        return loss

    def test_step(self, batch, batch_idx):
        """
        测试步骤 - 最终模型评估
        测试集评估，使用完整精度的指标计算
        """
        image, mask = batch
        with torch.no_grad():
            pred_mask = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_mask, mask)
            self.log(f"{self.dataset_name}_test_loss", loss.item())

            # 测试阶段使用完整精度的指标计算
            metrics = self.evaluator.estimate_metrics(
                pred_mask.sigmoid().squeeze(),
                mask.squeeze(),
                threshold=self.prediction_threshold  # 不使用快速模式
            )

            # 记录测试指标
            for name, value in metrics.items():
                value = value.item() if isinstance(value, (torch.Tensor, np.ndarray)) else value
                self.log(f"{self.dataset_name}_test_{name}", value)

        return loss