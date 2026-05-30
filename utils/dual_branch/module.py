import torch
import numpy as np
from lightning.pytorch import LightningModule
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import logging

logger = logging.getLogger(__name__)


class DualBranchPLModule(LightningModule):
    def __init__(self, model, dual_loss_fn, evaluator, dataset_name, optimizer_config, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = dual_loss_fn
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.optimizer_config = optimizer_config

        # 🌟 剔除 cldice 缓存，只保留极速的 DiceMetric
        self.val_dice = DiceMetric(include_background=False, reduction="mean")
        self.threshold = 0.5
        self.best_val_dice = 0.0

        # 🌟 护盾 1：新增步数计数器，用于精准识别断点碎片
        self.val_step_count = 0

        self.save_hyperparameters(ignore=['model', 'dual_loss_fn', 'evaluator'])

    def training_step(self, batch, batch_idx):
        batch_l = batch['labeled']
        batch_u = batch['unlabeled']

        img_l, mask_l = self._unpack_batch(batch_l)
        img_u, _ = self._unpack_batch(batch_u)

        # 🌟 极简前向传播，直接输出预测图 (不再请求特征)
        preds_l = self.model(img_l)
        preds_u = self.model(img_u)

        # 🌟 极简 Loss 计算，只接收 3 个返回值
        loss, loss_sup, loss_ps = self.loss_fn(
            preds_l=preds_l,
            mask_l=mask_l,
            preds_u=preds_u,
            current_epoch=self.current_epoch,
            img_l=img_l
        )

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("loss_sup", loss_sup, prog_bar=False, sync_dist=True)
        self.log("loss_ps", loss_ps, prog_bar=False, sync_dist=True)

        return loss

    # === 让控制台打印的历史最高分也能在断点后恢复 ===
    def on_save_checkpoint(self, checkpoint):
        checkpoint['best_val_dice_memory'] = self.best_val_dice

    def on_load_checkpoint(self, checkpoint):
        if 'best_val_dice_memory' in checkpoint:
            self.best_val_dice = checkpoint['best_val_dice_memory']

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch['Image'], batch['Mask']
        elif isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        else:
            raise TypeError(f"不支持的 Batch 类型: {type(batch)}")

    def validation_step(self, batch, batch_idx):
        val_inputs, gt_masks = self._unpack_batch(batch)

        val_outputs = sliding_window_inference(
            inputs=val_inputs,
            roi_size=(128, 128, 128),
            sw_batch_size=4,
            predictor=self.model,
            overlap=0.5
        )

        val_preds = (torch.sigmoid(val_outputs) > self.threshold).float()

        # 直接缓存 Dice
        self.val_dice(y_pred=val_preds, y=gt_masks)

        # 🌟 护盾 2：每次验证严格记录步数
        self.val_step_count += 1

    def on_validation_epoch_end(self):
        # 提取平均 Dice
        try:
            dice_score = self.val_dice.aggregate().item()
        except Exception:
            dice_score = 0.0

        # 清空缓存
        self.val_dice.reset()

        # 🌟 护盾 3：获取验证集预期的总 Batch 数量
        total_val_batches = sum(self.trainer.num_val_batches) if self.trainer.num_val_batches else 0

        # 判断是否为完整验证（排除了断点碎片，或者 Lightning 的 sanity check 阶段）
        is_fragment = self.val_step_count < total_val_batches

        if not is_fragment:
            # ==========================================
            # ✅ 完整评估：正常上报，允许覆盖 Best 记录
            # ==========================================
            is_best = dice_score > self.best_val_dice
            if is_best:
                self.best_val_dice = dice_score

            if self.trainer.is_global_zero:
                logger.info("\n" + "=" * 50)
                logger.info(f"📊 [Epoch {self.current_epoch}] 核心指标验证报告")
                logger.info(f"   🔹 Val Dice    : {dice_score * 100:.2f}%")
                logger.info(f"   🏆 Best Dice   : {self.best_val_dice * 100:.2f}%")
                logger.info("=" * 50 + "\n")

            # 正常上报分数给 Lightning 的监控系统
            self.log("val_DiceMetric", dice_score, prog_bar=True, sync_dist=True)

        else:
            # ==========================================
            # 🛑 碎片评估：触发安全拦截机制！
            # ==========================================
            if self.trainer.is_global_zero:
                logger.info("\n" + "=" * 50)
                logger.info("⚠️ [防误判拦截] 检测到极少样本的验证碎片 (通常由断点重连或模型自检产生)。")
                logger.info(f"⚠️ 当前测得不纯分数: {dice_score * 100:.2f}%，已将其隔离！")
                logger.info(f"   🏆 历史真实 Best 保持: {self.best_val_dice * 100:.2f}%")
                logger.info("=" * 50 + "\n")

            # 🌟 神来之笔：强行向系统上报 0.0 分。
            # 这样无论 ModelCheckpoint 怎么监控，都不会把这个垃圾轮次存成 Best
            self.log("val_DiceMetric", 0.0, prog_bar=True, sync_dist=True)

        # 🌟 护盾 4：无论如何，结束后必须将计数器归零，迎接下一轮
        self.val_step_count = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}