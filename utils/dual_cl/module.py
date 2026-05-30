import logging

import torch
from lightning.pytorch import LightningModule
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

logger = logging.getLogger(__name__)


class DualBranchPLModule(LightningModule):
    """
    对比学习版本的 LightningModule 封装。

    这个类本身不实现新的网络结构，也不直接实现损失细节，而是把下面三部分
    统一接入 Lightning 的标准训练流程：
      1. utils.dual_cl.model.DualStreamDynUNet
      2. utils.dual_cl.loss*.DualBranchLoss
      3. optimizer / scheduler / validation Dice

    和基础 dual_branch.module 的主要区别是：
      - 训练阶段需要额外拿到 decoder feature，用于小框对比学习；
      - loss_fn 会返回 4 个值：总损失、监督损失、伪标签损失、对比学习损失。

    训练阶段模型返回：
        preds, feats = model(image, return_features=True)

    损失函数返回：
        total_loss, supervised_loss, pseudo_loss, contrastive_loss
    """

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
        self.val_step_count = 0

        self.save_hyperparameters(ignore=["model", "dual_loss_fn", "evaluator"])

    def training_step(self, batch, batch_idx):
        """
        单个训练 step。

        batch 来自 DualStreamDataset，结构为：
            batch["labeled"]   -> 有切片弱标注样本
            batch["unlabeled"] -> 无标签样本

        当前 step 的逻辑：
          1. labeled 图像进入双分支模型，得到两个预测和两个特征图；
          2. unlabeled 图像进入双分支模型，得到两个预测和两个特征图；
          3. DualBranchLoss 同时计算监督损失、CPS 伪标签损失、对比学习损失；
          4. Lightning 自动对 total loss 反向传播。
        """
        img_l, mask_l = self._unpack_batch(batch["labeled"])
        img_u, _ = self._unpack_batch(batch["unlabeled"])

        # return_features=True 时，模型会返回：
        #   preds_l = (pred1_l, pred2_l)
        #   feats_l = (feat1_l, feat2_l)
        # 两个分支的 feature 后续用于小框对比学习。
        preds_l, feats_l = self.model(img_l, return_features=True)
        preds_u, feats_u = self.model(img_u, return_features=True)

        loss, loss_sup, loss_ps, loss_cl = self.loss_fn(
            preds_l=preds_l,
            mask_l=mask_l,
            preds_u=preds_u,
            feats_l=feats_l,
            feats_u=feats_u,
            current_epoch=self.current_epoch,
            img_l=img_l,
        )

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("loss_sup", loss_sup, prog_bar=False, sync_dist=True)
        self.log("loss_ps", loss_ps, prog_bar=False, sync_dist=True)
        if getattr(self.loss_fn, "enable_cl", False):
            self.log("loss_cl", loss_cl, prog_bar=True, sync_dist=True)
        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_val_dice_memory"] = self.best_val_dice

    def on_load_checkpoint(self, checkpoint):
        if "best_val_dice_memory" in checkpoint:
            self.best_val_dice = checkpoint["best_val_dice_memory"]

    def _unpack_batch(self, batch):
        """兼容 dict 和 tuple/list 两种 Dataset 返回格式。"""
        if isinstance(batch, dict):
            return batch["Image"], batch["Mask"]
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def validation_step(self, batch, batch_idx):
        """
        验证阶段只使用主分支/基础模型输出。

        DualStreamDynUNet 在 eval 模式下会直接返回 base_model(x)，不会返回 tuple，
        因此可以直接接入 MONAI sliding_window_inference。
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

        这里保留“碎片验证”保护：如果断点恢复或异常流程导致验证 batch 数少于
        Lightning 记录的完整验证 batch 数，则不允许该结果刷新 best checkpoint。
        """
        try:
            dice_score = self.val_dice.aggregate().item()
        except Exception:
            dice_score = 0.0
        self.val_dice.reset()

        total_val_batches = sum(self.trainer.num_val_batches) if self.trainer.num_val_batches else 0
        is_fragment = self.val_step_count < total_val_batches

        if not is_fragment:
            if dice_score > self.best_val_dice:
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

            self.log("val_DiceMetric", 0.0, prog_bar=True, sync_dist=True)

        self.val_step_count = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
