import torch
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

        self.val_dice = DiceMetric(include_background=False, reduction="mean")
        self.threshold = 0.5
        self.best_val_dice = 0.0

        # 🌟 护盾 1：新增步数计数器，用于识别断点碎片
        self.val_step_count = 0

        self.save_hyperparameters(ignore=['model', 'dual_loss_fn', 'evaluator'])

    def training_step(self, batch, batch_idx):
        img_l, mask_l = self._unpack_batch(batch['labeled'])
        img_u, _ = self._unpack_batch(batch['unlabeled'])

        # 获取预测图 和 特征图
        preds_l, feats_l = self.model(img_l, return_features=True)
        preds_u, feats_u = self.model(img_u, return_features=True)

        loss, loss_sup, loss_ps, loss_cl = self.loss_fn(
            preds_l=preds_l, mask_l=mask_l,
            preds_u=preds_u, feats_l=feats_l, feats_u=feats_u,
            current_epoch=self.current_epoch,
            img_l=img_l
        )

        # 记得 detach() 避免慢性内存泄漏
        self.log("train_loss", loss.detach(), prog_bar=True, sync_dist=True)
        self.log("loss_sup", loss_sup.detach(), prog_bar=False, sync_dist=True)
        self.log("loss_ps", loss_ps.detach(), prog_bar=False, sync_dist=True)
        if hasattr(self.loss_fn, 'enable_cl') and self.loss_fn.enable_cl:
            self.log("loss_cl", loss_cl.detach(), prog_bar=True, sync_dist=True)

        return loss

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

    # =========================================================================
    # 🌟 护盾 2：每次验证开启前，清零步数
    # =========================================================================
    def on_validation_epoch_start(self):
        self.val_step_count = 0

    def validation_step(self, batch, batch_idx):
        # 🌟 护盾 3：每验证一张图（或一个batch），步数 +1
        self.val_step_count += 1

        val_inputs, gt_masks = self._unpack_batch(batch)
        val_outputs = sliding_window_inference(
            inputs=val_inputs, roi_size=(128, 128, 128),
            sw_batch_size=4, predictor=self.model, overlap=0.5
        )
        val_preds = (torch.sigmoid(val_outputs) > self.threshold).float()
        self.val_dice(y_pred=val_preds, y=gt_masks)

    def on_validation_epoch_end(self):
        try:
            dice_score = self.val_dice.aggregate().item()
        except Exception:
            dice_score = 0.0
        self.val_dice.reset()

        # =========================================================================
        # 🌟 护盾 4：最核心的安全拦截逻辑
        # 判断本次验证的步数。如果少于等于 1 步，说明这绝不是完整验证，而是断点遗留的碎片！
        # =========================================================================
        is_valid_eval = self.val_step_count > 1  # 正常验证肯定大于 1 个 Batch

        if is_valid_eval:
            # 正常完整验证流程
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
            # 碎片评估，触发安全拦截机制！
            if self.trainer.is_global_zero:
                logger.info("\n" + "=" * 50)
                logger.info("⚠️ [防误判拦截] 检测到极少样本的验证碎片 (通常由断点重连产生)。")
                logger.info(f"⚠️ 当前测得异常分数: {dice_score * 100:.2f}%，已将其安全隔离！")
                logger.info(f"   🏆 历史真实 Best 保持: {self.best_val_dice * 100:.2f}%")
                logger.info("=" * 50 + "\n")

            # 🌟 神来之笔：强行向系统上报 0.0 分。
            # 这会让底层的 ModelCheckpoint 认为分数很烂，绝对不会触发权重覆盖操作！
            self.log("val_DiceMetric", 0.0, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_config.lr,
                                      weight_decay=self.optimizer_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}