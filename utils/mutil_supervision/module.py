import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric


class WeakPLModule(LightningModule):
    """
    PyTorch Lightning 核心模块 (LightningModule)

    作用：将 PyTorch 原本繁琐的 for 循环训练代码（如 zero_grad, backward, step）全部封装。
    你只需要定义“一次训练步(training_step)”和“一次验证步(validation_step)”，
    Lightning Trainer 会自动帮你处理多 GPU 分布式训练 (DDP)、半精度 (AMP)、以及设备转移 (to('cuda'))。
    """

    def __init__(
            self, model, loss, evaluator=None, dataset_name="imageCAS",
            prediction_threshold=0.5, optimizer_config=None, enable_post_process=False,
            **kwargs
    ):
        super().__init__()
        # 1. 挂载核心组件
        self.model = model  # 你的 3D 分割网络 (如 DynUNet)
        self.loss = loss  # 之前定义的损失函数 (如 FullLabelLoss 或 SparseSliceLoss)
        self.evaluator = evaluator  # 自定义的复杂评估器 (计算 clDice, Betti 数等)
        self.dataset_name = dataset_name
        self.threshold = prediction_threshold  # 将连续的概率值 (0~1) 截断为离散的二值标签 (0或1) 的阈值
        self.optimizer_config = optimizer_config
        self.enable_post_process = enable_post_process

        # 2. 定义 MONAI 原生的 Dice 评估器
        # 【极其重要】include_background=False：在血管分割中，背景体素可能占整个 3D 图像的 99% 以上。
        # 如果把背景也算进去，Dice 随便都能跑到 0.99，这就成了“虚假的高分”。
        # 关闭它，强迫模型只计算前景（血管）的交并比，反映真实的分割水平。
        # reduction="mean" 表示对整个 Batch 的 Dice 求平均。
        self.val_dice = DiceMetric(include_background=False, reduction="mean")

    def _unpack_batch(self, batch):
        """
        辅助函数：拆包 DataLoader 传过来的 Batch 数据。
        兼容 MONAI 基于字典的数据增强 (Dict Transform) 和普通的元组数据。
        """
        if isinstance(batch, dict):
            return batch['image'], batch['label']
        elif isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        else:
            raise TypeError(f"Batch type {type(batch)} not supported.")

    def training_step(self, batch, batch_idx):
        """
        定义一次前向训练的逻辑
        """
        # 1. 获取图像和掩膜
        images, masks = self._unpack_batch(batch)
        # 2. 前向传播
        # 输出的 logits 是网络最后一层尚未经过 Sigmoid 的原始数值
        logits = self.model(images)
        # 3. 计算损失
        # 注意：这里我们把 images 也传进去了。虽然全监督 Loss 用不到原图，
        # 但如果是切片弱监督 (SliceLoss)，它需要利用原图的三维空间灰度相似性(Affinity)来引导伪标签生长。
        loss_seg = self.loss(logits, masks, images)
        # 4. 日志记录
        # sync_dist=True: 在多 GPU 分布式训练 (DDP) 时，必须开启此项，
        # 它会自动把多张显卡上的 loss 收集起来求平均后再记录，保证指标的准确性。
        self.log("train_loss", loss_seg, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # 返回 loss 给内部的自动求导引擎执行 backward()
        return loss_seg

    def validation_step(self, batch, batch_idx):
        """
        定义一次验证评估的逻辑 (通常在整个 Epoch 结束后触发)
        """
        images, gt_masks = self._unpack_batch(batch)
        # =====================================================================
        # 🔥 滑动窗口推理 (Sliding Window Inference) - 3D 医疗影像的核心技巧 🔥
        # 3D 图像（如 512x512x256）通常太大，直接塞进 GPU 会直接 OOM (显存溢出)。
        # 训练时我们靠 RandomCrop 切割 128x128x128 的小块，但验证时我们需要整图的预测结果。
        #
        # 这个函数会：
        # 1. 在大图上以 128x128x128 的窗口滑动切割。
        # 2. 把每个小窗口送进 self.model 预测。
        # 3. 按照 overlap=0.5 (50% 的重叠率) 将这些小预测块重新“无缝拼接”回原始大尺寸。
        # 4. mode="gaussian"：在拼接重叠区域时，由于边缘通常预测不准，使用高斯加权，越靠近窗口中心的预测结果权重越高。
        # =====================================================================
        val_outputs = sliding_window_inference(images, (128, 128, 128), 1, self.model, overlap=0.5, mode="gaussian")
        # 兼容部分网络可能会返回多尺度 Tuple 的情况
        if isinstance(val_outputs, tuple):
            val_outputs = val_outputs[0]

        # 将连续的 Logits 转换为真实的 0 和 1 掩膜
        val_preds = (torch.sigmoid(val_outputs) > self.threshold).float()

        # 将这一步的预测结果送入 Dice 评估器中缓存累加 (不直接计算最终结果，等整个 epoch 结束一起算)
        self.val_dice(y_pred=val_preds, y=gt_masks)

    def on_validation_epoch_end(self):
        """
        当所有验证集数据都测试完毕后触发
        """
        # 1. 汇总整个验证集所有 batch 的缓存数据，计算出最终的平均 Dice 分数
        dice_score = self.val_dice.aggregate().item()

        # 2. 清空缓存，为下一个 Epoch 的验证做准备
        self.val_dice.reset()

        # 3. 记录最终指标。'val_DiceMetric' 这个名字会被 yaml 配置里的 ModelCheckpoint 监控，用来保存 best_model.ckpt
        self.log("val_DiceMetric", dice_score, prog_bar=True, sync_dist=True)
        self.log(f"{self.dataset_name}_val_dice", dice_score, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度策略
        """
        # AdamW 优化器：相比普通 Adam，它对 Weight Decay (权重衰减/L2正则化) 的实现更严谨，
        # 在医学图像这种容易过拟合的小样本数据集上表现更好。
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer_config['lr'],
                                      weight_decay=1e-5)

        # 余弦退火学习率调度器 (Cosine Annealing)
        # T_max 指定了最大的训练 Epoch 数。学习率会像半段余弦波浪一样，
        # 从初始的 1e-4 极其平滑地衰减到 0，有助于模型在训练后期收敛到更优的极小值。
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.trainer.max_epochs)

        # 按照 Lightning 的标准格式返回字典
        return {"optimizer": optimizer, "lr_scheduler": scheduler}