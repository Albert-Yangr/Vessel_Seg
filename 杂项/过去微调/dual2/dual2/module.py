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
        # 🌟 获取配置文件中是否启用了对比学习
        self.use_contrastive = kwargs.get("loss_configs", {}).get("contrastive", {}).get("enable", False)

        self.val_dice = DiceMetric(include_background=False, reduction="mean")
        self.val_cldice_outputs = []
        self.threshold = 0.5
        self.best_val_dice = 0.0
        self.save_hyperparameters(ignore=['model', 'dual_loss_fn', 'evaluator'])

    def training_step(self, batch, batch_idx):
        batch_l = batch['labeled']
        batch_u = batch['unlabeled']

        img_l, mask_l = self._unpack_batch(batch_l)
        img_u, _ = self._unpack_batch(batch_u)

        # 🌟 根据开关决定是否请求特征
        if self.use_contrastive:
            preds1_l, preds2_l, feat1_l, feat2_l = self.model(img_l, return_features=True)
            preds1_u, preds2_u, feat1_u, feat2_u = self.model(img_u, return_features=True)
            preds_l = (preds1_l, preds2_l)
            preds_u = (preds1_u, preds2_u)
            feats_l = (feat1_l, feat2_l)
            feats_u = (feat1_u, feat2_u)
        else:
            preds_l = self.model(img_l)
            preds_u = self.model(img_u)
            feats_l, feats_u = None, None

        loss, loss_sup, loss_ps, loss_cl = self.loss_fn(
            preds_l=preds_l,
            mask_l=mask_l,
            preds_u=preds_u,
            current_epoch=self.current_epoch,
            img_l=img_l,
            feats_l=feats_l,  # 🌟 传入特征
            feats_u=feats_u
        )

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("loss_sup", loss_sup, prog_bar=False, sync_dist=True)
        self.log("loss_ps", loss_ps, prog_bar=False, sync_dist=True)
        if loss_cl > 0:
            self.log("loss_cl", loss_cl, prog_bar=False, sync_dist=True)

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
        self.val_dice(y_pred=val_preds, y=gt_masks)

        val_preds_np = val_preds.cpu().numpy()
        gt_masks_np = gt_masks.cpu().numpy()

        for i in range(val_preds_np.shape[0]):
            pred_3d = val_preds_np[i, 0].astype(np.uint8)
            gt_3d = gt_masks_np[i, 0].astype(np.uint8)
            cldice_score = self.evaluator.cl_dice(v_p=pred_3d, v_l=gt_3d)
            self.val_cldice_outputs.append(cldice_score)

    def on_validation_epoch_end(self):
        # ==========================================================
        # 🌟 换种方法：基于真实处理的样本数量进行硬核防伪拦截！
        # ==========================================================
        sample_count = len(self.val_cldice_outputs)

        if sample_count < 2:
            # 🚨 如果发现只评估了 1 个（或0个）样本，强制将得分设为 0.0！
            # 这样就算个例简单考了满分，也绝对无法去覆盖你的历史真实权重！
            dice_score = 0.0
            cldice_score = 0.0
            if self.trainer.is_global_zero:
                logger.warning(f"\n🚨 拦截到异常/单样本评估 (仅处理了 {sample_count} 个数据)！")
                logger.warning("🚨 已强制将本轮得分归零 (0.0)，彻底掐死虚假高分覆盖权重的可能！")
        else:
            # 正常的完整评估
            try:
                dice_score = self.val_dice.aggregate().item()
            except Exception:
                dice_score = 0.0
            cldice_score = float(np.mean(self.val_cldice_outputs))

        # 清空缓存
        self.val_dice.reset()
        self.val_cldice_outputs.clear()

        # 更新最高分记录
        is_best = dice_score > self.best_val_dice
        if is_best:
            self.best_val_dice = dice_score

        # 打印日志
        if self.trainer.is_global_zero:
            logger.info("\n" + "=" * 50)
            if sample_count < 2:
                logger.info("⚠️ 本次异常评估已被安全阻断，不计入排名。")
            else:
                logger.info(f"📊 [Epoch {self.current_epoch}] 核心指标验证报告")
                logger.info(f"   🔹 Val Dice    : {dice_score * 100:.2f}%")
                logger.info(f"   🔹 Val clDice  : {cldice_score * 100:.2f}%")
                logger.info(f"   🏆 Best Dice   : {self.best_val_dice * 100:.2f}%")
            logger.info("=" * 50 + "\n")

        # 🌟 必须将分数（哪怕是拦截后的0.0）上报，满足Lightning不报错
        self.log("val_DiceMetric", dice_score, prog_bar=True, sync_dist=True)
        self.log("val_clDice", cldice_score, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}