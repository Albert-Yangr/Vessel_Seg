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

from utils.TASA.tasa_unet import TASA_VesselNet
from utils.TASA.cldice_loss import clDiceLoss, soft_skeletonize
from utils.TASA.clce_loss import clCELoss
from monai.inferers import sliding_window_inference

sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


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
            logger.info(f"⭐ 历史最佳 Dice: {best:.4f} | {icon}")
        logger.info("=" * 60)


class TopologyLossWrapper(torch.nn.Module):
    def __init__(self, base_loss, topo_name="clce", topo_weight=0.5):
        super().__init__()
        self.base_loss = base_loss
        self.topo_name = topo_name.lower()
        self.topo_weight = topo_weight
        self.cldice_loss = clDiceLoss(iters=5)
        self.clce_loss = clCELoss(iters=5)
        self.weights = (0.5, 0.3, 0.2)

    def forward(self, preds, target):
        if isinstance(preds, dict) and "vol" in preds and "skel" in preds:
            main_preds = preds["vol"]
            skel_preds = preds["skel"]
        else:
            main_preds = preds if isinstance(preds, (list, tuple)) else [preds]
            skel_preds = []
            print("\n[🚨 警告]: 传入 Loss 的格式非 Dict，双任务流脱轨！\n")

        total_loss = 0.0

        # =======================================================
        # 1. 主分支损失 (Volume Loss)
        # =======================================================
        for i, p in enumerate(main_preds):
            if i >= len(self.weights): break
            t = F.interpolate(target.float(), size=p.shape[2:], mode='nearest')
            loss_vol = self.base_loss(p, t)

            if i == 0 and self.topo_weight > 0:
                if self.topo_name == "cldice":
                    loss_topo = self.cldice_loss(torch.sigmoid(p), t)
                elif self.topo_name == "clce":
                    loss_topo = self.clce_loss(p, t)
                else:
                    loss_topo = 0.0
                loss_vol += self.topo_weight * loss_topo

            # 🔥 修复：如果基础模型只有单路输出，权重必须为 1.0
            w = self.weights[i] if len(main_preds) > 1 else 1.0
            total_loss += w * loss_vol

        # =======================================================
        # 2. 拓扑分支专属监督 (Skeleton Loss - 解决严重不平衡与坍塌)
        # =======================================================
        if len(skel_preds) > 0:
            target_skel = soft_skeletonize(target.float(), iters=5).detach()
            target_skel_bin = (target_skel > 0.1).float()

            for p_skel in skel_preds:
                target_shape = p_skel.shape[2:]

                # 尺度感知：防止深层特征物理坍塌引发数学噪声
                if target_shape[0] < target.shape[2] // 4:
                    continue

                t_s = F.adaptive_max_pool3d(target_skel_bin, output_size=target_shape)
                p_skel_prob = torch.sigmoid(p_skel)

                inter = (p_skel_prob * t_s).sum()
                union = p_skel_prob.sum() + t_s.sum()
                loss_skel_dice = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)
                loss_skel_bce = F.binary_cross_entropy_with_logits(p_skel, t_s)

                total_loss += 0.5 * (loss_skel_dice + loss_skel_bce)

        return total_loss


@hydra.main(config_path="../configs", config_name="train/train_tasa", version_base="1.3")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    rank = int(os.environ.get("RANK", 0))

    d_name = list(cfg.data.keys())[0]
    expert_type = cfg.get("expert_type", "snake")

    # 🔥 智能判断是单一专家还是组合混用，进行命名
    if isinstance(expert_type, str):
        exp_suffix = expert_type.upper()
        log_exp_str = exp_suffix
    else:
        exp_suffix = "MIXED"
        log_exp_str = " | ".join([e.upper() for e in list(expert_type)])

    run_name = f'{cfg.loss_name}_{d_name}_{exp_suffix}'
    experiment_dir = os.path.join(f"../local_results/doc/{d_name}", run_name)
    if rank == 0: os.makedirs(experiment_dir, exist_ok=True)

    topo_name = cfg.get("topo_loss", {}).get("name", "clce").lower()
    topo_weight = cfg.get("topo_loss", {}).get("weight", 0.5)

    if rank == 0:
        logger.info(f"📋 [训练策略]: 拓扑骨架适配器 (TASA) + 双流任务解耦")
        logger.info(f"🧠 [模块配置]: 挂载拓扑专家 -> {log_exp_str}")
        logger.info(f"🧬 [损失配置]: 使用 {topo_name.upper()} Loss, 惩罚权重: {topo_weight}")

    loggers = [
        WandbLogger(save_dir=experiment_dir, name=run_name, config=OmegaConf.to_container(cfg), offline=True),
        CleanCSVLogger(save_dir=experiment_dir, name="", version="")
    ]

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{cfg.chkpt_folder}/{d_name}/{run_name}",
        monitor="val_DiceMetric",
        mode="max",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        filename="TASA-SOTA-{epoch:02d}-{val_DiceMetric:.4f}",
        auto_insert_metric_name=False
    )

    train_dataset = UnionDataset(cfg.data, "train", finetune=True, repeats=cfg.get("repeats", 1))
    val_dataset = UnionDataset(cfg.data, "test", finetune=True, repeats=1)

    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset, batch_size=cfg.batch_size,
                                                           shuffle=True, pin_memory=True, drop_last=True)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1, shuffle=False)

    # =========================================================================
    # 🔥 核心修改：模型加载与“寄生”机制
    # =========================================================================

    # 1. 实例化官方的纯净基础模型
    base_model = hydra.utils.instantiate(cfg.model)

    # 2. 原装加载预训练权重，保证 100% 吻合
    if rank == 0: logger.info(f"🔄 正在原装加载基础模型权重: {cfg.path_to_chkpt}")
    if cfg.path_to_chkpt:
        ckpt = torch.load(cfg.path_to_chkpt, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        state = {k.replace('model.', '').replace('net.', ''): v for k, v in state.items()}
        # 严丝合缝加载
        base_model.load_state_dict({k: v for k, v in state.items() if k in base_model.state_dict()}, strict=False)
        if rank == 0: logger.info(f"✅ 权重 100% 完美贴合加载！")

    # 3. 将基础大模型包裹入 TASA 寄生壳中
    model = TASA_VesselNet(base_model=base_model, expert_type=expert_type)


    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"🔥 TASA 寄生机甲已挂载！拓扑可训练参数量: {trainable_params / 1e6:.2f} M")

    # =========================================================================

    base_loss = hydra.utils.instantiate(cfg.loss_configs.label_loss)
    topo_loss = TopologyLossWrapper(base_loss=base_loss, topo_name=topo_name, topo_weight=topo_weight)

    pl_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, evaluator=Evaluator(), dataset_name=d_name, loss=topo_loss
    )

    original_validation_step = pl_module.validation_step

    def custom_validation_step(self, batch, batch_idx):
        ret = original_validation_step(batch, batch_idx)
        if isinstance(batch, dict):
            images, labels = batch["image"], batch["label"]
        else:
            images, labels = batch[0], batch[1]

        val_outputs = sliding_window_inference(images, (128, 128, 128), 1, self.model, overlap=0.5, mode="gaussian")
        pred_mask = (torch.sigmoid(val_outputs) > 0.5).float()

        skel_pred = soft_skeletonize(pred_mask, iters=5)
        skel_true = soft_skeletonize(labels.float(), iters=5)

        smooth = 1e-5
        tprec = (torch.sum(skel_pred * labels.float()) + smooth) / (torch.sum(skel_pred) + smooth)
        tsens = (torch.sum(skel_true * pred_mask) + smooth) / (torch.sum(skel_true) + smooth)
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)

        self.log(f"{self.dataset_name}_val_cldice", cl_dice, sync_dist=True, on_step=False, on_epoch=True)
        return ret

    pl_module.validation_step = types.MethodType(custom_validation_step, pl_module)

    def custom_configure_optimizers(self):
        # 1. 确保整个网络的所有参数都处于解冻（可导）状态
        for param in self.model.parameters():
            param.requires_grad = True

        # 2. 参数分组容器
        encoder_params = []
        decoder_params = []
        tasa_params = []

        # 3. 剥离大模型 (DynUNet) 的 Encoder 和 Decoder 参数
        for name, param in self.model.base_model.named_parameters():
            # MONAI 的下采样链表层或 skip_layers 属于 Encoder
            if 'downsample' in name or 'skip_layers' in name:
                encoder_params.append(param)
            else:
                # 剩下的上采样层、输出头等属于 Decoder
                decoder_params.append(param)

        # 4. 剥离外挂的 TASA 拓扑专家参数
        for name, param in self.model.adapters.named_parameters():
            tasa_params.append(param)

        # 5. 读取 YAML 中的分层学习率倍率配置
        base_lr = cfg.lr
        enc_mult = cfg.get("lr_multipliers", {}).get("encoder", 0.1)
        dec_mult = cfg.get("lr_multipliers", {}).get("decoder", 1.0)
        exp_mult = cfg.get("lr_multipliers", {}).get("expert", 10.0)

        # 仅在主进程打印分层学习率信息
        if int(os.environ.get("RANK", 0)) == 0:
            logger.info(f"📈 [分层学习率] Encoder: {base_lr * enc_mult:.1e} | Decoder: {base_lr * dec_mult:.1e} | TASA: {base_lr * exp_mult:.1e}")

        # 6. 构建带有 Parameter Groups 的优化器
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': base_lr * enc_mult},
            {'params': decoder_params, 'lr': base_lr * dec_mult},
            {'params': tasa_params, 'lr': base_lr * exp_mult}
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.trainer.lightning_trainer.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    pl_module.configure_optimizers = types.MethodType(custom_configure_optimizers, pl_module)

    trainer = hydra.utils.instantiate(
        cfg.trainer.lightning_trainer,
        logger=loggers,
        callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
        devices=cfg.devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        sync_batchnorm=True,
        enable_model_summary=False,
    )()

    trainer.fit(pl_module, train_loader, val_loader)


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    main()