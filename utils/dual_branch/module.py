import logging

import torch
from lightning.pytorch import LightningModule
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

logger = logging.getLogger(__name__)


class DualBranchPLModule(LightningModule):
    """
    PyTorch Lightning 训练封装。

    这个类不定义新的网络结构，也不定义新的损失细节；它负责把：
        model.py 中的双分支模型
        loss.py 中的双分支损失
        optimizer / scheduler
        validation Dice
    接入 Lightning 的标准 training loop。

    Lightning 会自动调用：
        training_step()
        validation_step()
        on_validation_epoch_end()
        configure_optimizers()
    """

    def __init__(self, model, dual_loss_fn, evaluator, dataset_name, optimizer_config, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = dual_loss_fn
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.optimizer_config = optimizer_config

        # 验证阶段只统计前景 Dice。
        self.val_dice = DiceMetric(include_background=False, reduction="mean")
        self.threshold = 0.5
        self.best_val_dice = 0.0

        # 用于识别“断点恢复时产生的不完整验证片段”，避免错误刷新 best。
        self.val_step_count = 0

        self.save_hyperparameters(ignore=["model", "dual_loss_fn", "evaluator"])

    def training_step(self, batch, batch_idx):
        """
        单个训练 step。
        batch 来自 DualStreamDataset，结构为：
            batch["labeled"]   -> 有标签/弱监督样本
            batch["unlabeled"] -> 无标签样本
        训练逻辑：
            1. 有标签图像送入双分支模型，得到 preds_l=(pred1_l, pred2_l)。
            2. 无标签图像送入双分支模型，得到 preds_u=(pred1_u, pred2_u)。
            3. DualBranchLoss 计算 supervised loss + pseudo loss。
        """
        batch_l = batch["labeled"]
        batch_u = batch["unlabeled"]
        img_l, mask_l = self._unpack_batch(batch_l)
        img_u, _ = self._unpack_batch(batch_u)
        preds_l = self.model(img_l)
        preds_u = self.model(img_u)

        loss, loss_sup, loss_ps = self.loss_fn(
            preds_l=preds_l,
            mask_l=mask_l,
            preds_u=preds_u,
            current_epoch=self.current_epoch,
            img_l=img_l,
        )
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("loss_sup", loss_sup, prog_bar=False, sync_dist=True)
        self.log("loss_ps", loss_ps, prog_bar=False, sync_dist=True)

        return loss

    def on_save_checkpoint(self, checkpoint):
        """保存历史 best Dice，确保断点恢复后日志显示和 best 判断连续。"""
        checkpoint["best_val_dice_memory"] = self.best_val_dice

    def on_load_checkpoint(self, checkpoint):
        """从 checkpoint 中恢复历史 best Dice。"""
        if "best_val_dice_memory" in checkpoint:
            self.best_val_dice = checkpoint["best_val_dice_memory"]

    def _unpack_batch(self, batch):
        """
        兼容不同 Dataset 返回格式。

        dict 格式：
            {"Image": image, "Mask": mask}
        tuple/list 格式：
            (image, mask)
        """
        if isinstance(batch, dict):
            return batch["Image"], batch["Mask"]
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def validation_step(self, batch, batch_idx):
        """
        验证 step。
        验证阶段模型处于 eval 模式，因此 DualStreamDynUNet 只返回 base_model 的
        单分支输出。这里使用 MONAI sliding_window_inference 处理 3D 大图。
        """
        val_inputs, gt_masks = self._unpack_batch(batch)

        val_outputs = sliding_window_inference(
            inputs=val_inputs,
            roi_size=(128, 128, 128),
            sw_batch_size=4,
            predictor=self.model,
            overlap=0.5,
        )
        val_preds = (torch.sigmoid(val_outputs) > self.threshold).float()
        self.val_dice(y_pred=val_preds, y=gt_masks)
        self.val_step_count += 1

    def on_validation_epoch_end(self):
        """
        汇总一个验证周期的 Dice。

        这里额外判断验证是否完整，避免恢复断点或 sanity check 产生的极少量验证
        batch 错误刷新 best checkpoint。
        """
        try:
            dice_score = self.val_dice.aggregate().item()
        except Exception:
            dice_score = 0.0

        self.val_dice.reset()

        total_val_batches = sum(self.trainer.num_val_batches) if self.trainer.num_val_batches else 0
        is_fragment = self.val_step_count < total_val_batches

        if not is_fragment:
            is_best = dice_score > self.best_val_dice
            if is_best:
                self.best_val_dice = dice_score

            if self.trainer.is_global_zero:
                logger.info("\n" + "=" * 50)
                logger.info(f"[Epoch {self.current_epoch}] validation report")
                logger.info(f"  Val Dice : {dice_score * 100:.2f}%")
                logger.info(f"  Best Dice: {self.best_val_dice * 100:.2f}%")
                logger.info("=" * 50 + "\n")

            self.log("val_DiceMetric", dice_score, prog_bar=True, sync_dist=True)
        else:
            if self.trainer.is_global_zero:
                logger.info("\n" + "=" * 50)
                logger.info("[Validation skipped] detected fragmented validation after resume/sanity check.")
                logger.info(f"  Fragment Dice ignored: {dice_score * 100:.2f}%")
                logger.info(f"  Historical Best kept: {self.best_val_dice * 100:.2f}%")
                logger.info("=" * 50 + "\n")

            # 上报 0，避免 ModelCheckpoint 误保存不完整验证结果。
            self.log("val_DiceMetric", 0.0, prog_bar=True, sync_dist=True)

        self.val_step_count = 0

    def configure_optimizers(self):
        """
        优化器和学习率调度。
        当前使用 AdamW + CosineAnnealingLR。
        lr 和 weight_decay 来自 dual_train.yaml 的 optimizer 配置。
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
