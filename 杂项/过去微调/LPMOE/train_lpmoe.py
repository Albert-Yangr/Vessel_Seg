import logging
import sys
import warnings
import os
import hydra
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf, open_dict
import types

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from utils.full_train_dataset import UnionDataset
from utils.evaluation import Evaluator
from utils.LPMoE.lpmoe_unet import LPMoE_VesselNet

# 🌟 引入核心拓扑约束
from utils.TASA.cldice_loss import clDiceLoss, soft_skeletonize
from utils.TASA.clce_loss import clCELoss
from monai.inferers import sliding_window_inference
import numpy as np
from skimage.morphology import skeletonize_3d
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def get_hard_skeleton_cpu(masks_tensor):
    """在CPU上实时生成精确硬骨架，零GPU显存消耗"""
    # 转移到 CPU 并转为 bool 型 numpy
    masks_np = masks_tensor.detach().cpu().numpy() > 0.5
    skels_np = np.zeros_like(masks_np, dtype=np.float32)

    # 逐个 batch 提取精准 1 像素骨架
    for b in range(masks_np.shape[0]):
        skels_np[b, 0] = skeletonize_3d(masks_np[b, 0])

    # 送回原来的 GPU 节点
    return torch.from_numpy(skels_np).to(masks_tensor.device)
class CleanCSVLogger(CSVLogger):
    def log_hyperparams(self, params): pass


class LogCallback(LearningRateMonitor):
    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0: return
        m = trainer.callback_metrics
        d_name = pl_module.dataset_name
        epoch = trainer.current_epoch

        score = m.get(f"{d_name}_val_dice") or m.get("val_DiceMetric")
        cldice = m.get(f"{d_name}_val_cldice")
        best = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None

        logger.info(f"{'=' * 15} Epoch {epoch} {'=' * 15}")
        if score: logger.info(f"✅ 验证 Dice:   {score:.4f}")
        if cldice: logger.info(f"🧬 验证 clDice: {cldice:.4f} (拓扑连通性)")
        if best and score:
            diff = float(score) - float(best)
            icon = f"🚀 新纪录! (+{diff:.4f})" if diff > 1e-6 else (
                "⚖️  持平" if diff > -1e-6 else f"🔙 差距: {diff:.4f}")
            logger.info(f"⭐ 历史最佳: {best:.4f} | {icon}")
        logger.info("=" * 60)


# =========================================================================
# 🌟 核心创新的集大成者：深度监督拓扑损失中心
# 主分支保 Dice，旁路专家被强迫寻找骨架！
# =========================================================================
class TopologyLossWrapper(torch.nn.Module):
    def __init__(self, base_loss, topo_name="clce", topo_weight=0.5):
        super().__init__()
        self.base_loss = base_loss
        self.topo_name = topo_name.lower()
        self.topo_weight = topo_weight
        # 🔥 对预测值的软骨架化仍需保留(为了反向传播求导)，但迭代降为2，彻底告别爆显存
        self.cldice_loss = clDiceLoss(iters=2)
        self.clce_loss = clCELoss(iters=2)

    def forward(self, preds, target):
        if isinstance(preds, dict) and "vol" in preds and "skel" in preds:
            main_preds = preds["vol"]
            skel_preds = preds["skel"]
        else:
            main_preds = [preds] if not isinstance(preds, (list, tuple)) else preds
            skel_preds = []

        total_loss = 0.0

        # 1. 主分支损失 (Volume Loss)
        for p in main_preds:
            t = F.interpolate(target.float(), size=p.shape[2:], mode='nearest')
            loss_vol = self.base_loss(p, t)

            if self.topo_weight > 0:
                loss_topo = self.clce_loss(p, t) if self.topo_name == "clce" else self.cldice_loss(torch.sigmoid(p), t)
                loss_vol += self.topo_weight * loss_topo
            total_loss += loss_vol

        # 2. 专家专属特训 (Skeleton Loss)
        if len(skel_preds) > 0:
            # 🔥 核心杀招：使用 CPU 计算出绝对精确的 Target 骨架，0 显存占用！
            target_skel_bin = get_hard_skeleton_cpu(target)

            # 只监控前 3 层高分辨率的专家
            for p_skel in skel_preds[:3]:
                target_shape = p_skel.shape[2:]

                # 直接通过自适应池化将高分辨率骨架下采样给深层使用
                t_s = F.adaptive_max_pool3d(target_skel_bin, output_size=target_shape)
                p_skel_prob = torch.sigmoid(p_skel)

                inter = (p_skel_prob * t_s).sum()
                union = p_skel_prob.sum() + t_s.sum()
                loss_skel_dice = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)
                loss_skel_bce = F.binary_cross_entropy_with_logits(p_skel, t_s)

                total_loss += 0.5 * (loss_skel_dice + loss_skel_bce)

        return total_loss


def smart_load_weights(model, ckpt_path, rank=0):
    if not ckpt_path: return
    if rank == 0: logger.info(f"🔄 开始强力映射基础模型权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    state = {k.replace('model.', '').replace('net.', ''): v for k, v in state.items()}
    curr_state = model.state_dict()

    src_tensors = [v for k, v in state.items() if v.ndim >= 1 and 'tracked' not in k]
    tgt_keys = [k for k in curr_state.keys() if ('encoder' in k or 'decoder' in k or 'out_conv' in k) and curr_state[
        k].ndim >= 1 and 'tracked' not in k]

    loaded, src_idx = 0, 0
    for t_key in tgt_keys:
        t_tensor = curr_state[t_key]
        for i in range(src_idx, len(src_tensors)):
            if src_tensors[i].shape == t_tensor.shape:
                curr_state[t_key] = src_tensors[i].clone()
                src_idx = i + 1
                loaded += 1
                break
    model.load_state_dict(curr_state, strict=False)
    if rank == 0: logger.info(f"✅ 权重映射完成！恢复了 {loaded}/{len(tgt_keys)} 个骨干层。")


@hydra.main(config_path="../configs", config_name="train/train_lpmoe", version_base="1.3")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    rank = int(os.environ.get("RANK", 0))

    d_name = list(cfg.data.keys())[0]
    run_name = f'Topo_{cfg.loss_name}_{d_name}'
    experiment_dir = os.path.join(f"../local_results/doc/{d_name}", run_name)
    if rank == 0: os.makedirs(experiment_dir, exist_ok=True)

    if rank == 0: logger.info(f"📋 [训练模式]: 全量解冻 + Topo-LPMoE 混合专家拓扑分层微调")

    loggers = [
        WandbLogger(save_dir=experiment_dir, name=run_name, config=OmegaConf.to_container(cfg), offline=True),
        CleanCSVLogger(save_dir=experiment_dir, name="", version="")
    ]

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{cfg.chkpt_folder}/{d_name}/{run_name}",
        monitor="val_DiceMetric", mode="max", save_top_k=1, save_last=False, save_weights_only=True,
        filename="Topo-LPMoE-{epoch:02d}-{val_DiceMetric:.4f}", auto_insert_metric_name=False
    )

    train_dataset = UnionDataset(cfg.data, "train", finetune=True, repeats=cfg.get("repeats", 1))
    val_dataset = UnionDataset(cfg.data, "test", finetune=True, repeats=1)
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset, batch_size=cfg.batch_size,
                                                           shuffle=True, pin_memory=True, drop_last=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1, shuffle=False)

    model = LPMoE_VesselNet(spatial_dims=3, filters=(32, 64, 128, 256, 320, 320))
    smart_load_weights(model, cfg.path_to_chkpt, rank)

    # 🔥 我们不再冻结模型！
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"🔥 全量解冻模式激活！可训练参数量: {trainable_params / 1e6:.2f} M")

    base_loss = hydra.utils.instantiate(cfg.loss_configs.label_loss)
    topo_loss_name = cfg.get("topo_loss", {}).get("name", "clce")
    topo_weight = cfg.get("topo_loss", {}).get("weight", 0.5)

    topo_lpmoe_loss = TopologyLossWrapper(base_loss=base_loss, topo_name=topo_loss_name, topo_weight=topo_weight)

    pl_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, evaluator=Evaluator(), dataset_name=d_name, loss=topo_lpmoe_loss
    )

    original_validation_step = pl_module.validation_step

    def custom_validation_step(self, batch, batch_idx):
        ret = original_validation_step(batch, batch_idx)
        images, labels = (batch["image"], batch["label"]) if isinstance(batch, dict) else (batch[0], batch[1])
        val_outputs = sliding_window_inference(images, (128, 128, 128), 1, self.model, overlap=0.5, mode="gaussian")
        pred_mask = (torch.sigmoid(val_outputs) > 0.5).float()

        # 🔥 验证时也不要用 soft_skeletonize 浪费显存了，直接用 CPU 算最标准的硬骨架评估
        skel_pred = get_hard_skeleton_cpu(pred_mask)
        skel_true = get_hard_skeleton_cpu(labels)

        smooth = 1e-5
        tprec = (torch.sum(skel_pred * labels.float()) + smooth) / (torch.sum(skel_pred) + smooth)
        tsens = (torch.sum(skel_true * pred_mask) + smooth) / (torch.sum(skel_true) + smooth)
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
        self.log(f"{self.dataset_name}_val_cldice", cl_dice, sync_dist=True, on_step=False, on_epoch=True)
        return ret

    pl_module.validation_step = types.MethodType(custom_validation_step, pl_module)

    # =========================================================================
    # 🔥 重写的全量解冻 + 分层学习率优化器
    # =========================================================================
    def custom_configure_optimizers(self):
        for param in self.model.parameters(): param.requires_grad = True

        encoder_params, decoder_params, expert_params = [], [], []
        for name, param in self.model.named_parameters():
            if 'encoder_blocks' in name:
                encoder_params.append(param)
            elif 'decoder_blocks' in name or 'out_conv' in name:
                decoder_params.append(param)
            else:
                expert_params.append(param)

        base_lr = cfg.lr
        enc_mult = cfg.get("lr_multipliers", {}).get("encoder", 0.1)
        dec_mult = cfg.get("lr_multipliers", {}).get("decoder", 1.0)
        exp_mult = cfg.get("lr_multipliers", {}).get("expert", 10.0)

        if int(os.environ.get("RANK", 0)) == 0:
            logger.info(
                f"📈 [分层联合微调] Encoder: {base_lr * enc_mult:.1e} | Decoder: {base_lr * dec_mult:.1e} | Experts: {base_lr * exp_mult:.1e}")

        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': base_lr * enc_mult},
            {'params': decoder_params, 'lr': base_lr * dec_mult},
            {'params': expert_params, 'lr': base_lr * exp_mult}
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=cfg.trainer.lightning_trainer.max_epochs,
                                                               eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    pl_module.configure_optimizers = types.MethodType(custom_configure_optimizers, pl_module)

    trainer = hydra.utils.instantiate(
        cfg.trainer.lightning_trainer, logger=loggers, callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
        devices=cfg.devices, accelerator="gpu", strategy=DDPStrategy(find_unused_parameters=True),
        sync_batchnorm=True, enable_model_summary=False,
    )()
    trainer.fit(pl_module, train_loader, val_loader)


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()