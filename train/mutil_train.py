import logging
import sys
import warnings
import os
# 必须放在 import torch 和其他底层库之前！
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
import hydra
import torch
from pathlib import Path
from omegaconf import OmegaConf, open_dict




# 导入 PyTorch Lightning 核心组件
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy

# 引入基础组件（自定义的数据集和评估工具）
from utils.evaluation import Evaluator
from utils.mutil_supervision.mutil_dataset import UnionDataset, MutilSupervisionDataset

# =========================================================================
# 环境与全局配置设置
# =========================================================================

# 将当前脚本的父目录的父目录（即项目根目录）加入到系统路径中
# 这样可以确保在使用绝对路径导入（如 from utils...）时，Python 能正确找到对应模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 忽略烦人的警告信息，保持终端输出整洁
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# 调整底层库的日志级别：只在发生 ERROR（错误）时才打印信息，屏蔽默认的 INFO 刷屏
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("monai").setLevel(logging.ERROR)

# 尝试屏蔽 MONAI 在数据增强时可能触发的全局随机种子警告
try:
    import monai.transforms.transform

    # 将 MONAI 的最大种子值设为 32位无符号整数最大值，防止溢出警告
    monai.transforms.transform.MAX_SEED = 0xFFFFFFFF
except ImportError:
    pass

# 初始化当前模块的日志记录器，方便我们在控制台打印美观的训练信息
logger = logging.getLogger(__name__)


# =========================================================================
# 自定义组件 (回调函数 Callbacks & 工具 Utils)
# =========================================================================

class CleanCSVLogger(CSVLogger):
    """
    自定义的 CSV 记录器。
    重写了 log_hyperparams 方法并将其置空（pass），
    目的是为了防止由于 Hydra 传进来的复杂配置字典无法被安全序列化而导致的报错。
    这样它只会老老实实地把 Loss 和 Dice 等指标记录进 metrics.csv 中。
    """

    def log_hyperparams(self, params):
        pass


class LogCallback(LearningRateMonitor):
    """
    自定义日志回调函数，继承自 LearningRateMonitor (为了顺便记录学习率)。
    主要功能是在每个 Validation (验证) 周期结束时，在控制台打印排版精美的训练简报。
    """

    def on_validation_end(self, trainer, pl_module):
        # 多卡分布式训练时（DDP），只让主进程 (rank 0) 打印日志，防止多张卡重复打印
        if trainer.global_rank != 0: return

        # 获取当前所有的监控指标字典
        m = trainer.callback_metrics
        d_name = pl_module.dataset_name

        epoch = trainer.current_epoch
        step = trainer.global_step

        # 尝试获取验证集 Dice 分数（兼容不同的命名习惯）
        score = m.get(f"{d_name}_val_dice") or m.get("val_DiceMetric")
        # 尝试获取验证集和训练集的 Loss
        loss = m.get(f"{d_name}_val_loss") or m.get("val_loss")
        loss_train = m.get("train_loss")
        loss_cont = m.get("cont_loss")

        # 获取 ModelCheckpoint 回调中记录的历史最高分
        best = trainer.checkpoint_callback.best_model_score if trainer.checkpoint_callback else None

        # 打印华丽的分割线和当前进度
        logger.info(f"{'=' * 15} Epoch {epoch} | Step {step} {'=' * 15}")

        # 按条件格式化打印各项指标
        if score: logger.info(f"✅ 验证 Dice: {score:.4f}")
        if loss:  logger.info(f"📉 验证 Loss: {loss:.4f}")
        if loss_train: logger.info(f"🚂 训练 Loss: {loss_train:.4f}")
        if loss_cont:  logger.info(f"🆚 对比 Loss: {loss_cont:.4f}")

        # 智能对比当前分数与历史最佳分数，并给予对应的Emoji反馈
        if best and score:
            diff = float(score) - float(best)
            if diff > 1e-6:
                icon = f"🚀 新纪录! (+{diff:.4f})"
            elif diff > -1e-6:
                icon = "⚖️  持平"
            else:
                icon = f"🔙 差距: {diff:.4f}"
            logger.info(f"⭐ 历史最佳: {best:.4f} | {icon}")
        logger.info("=" * 60)


def safe_load_weights(model, ckpt_path, rank=0):
    """
    安全地加载预训练权重（如 Stage 1 基础模型的权重）。
    具备强大的容错能力，如果发现网络层结构有微小变动（如通道数不一致），
    它会自动过滤掉不兼容的层，只加载匹配的权重。
    """
    if not ckpt_path: return  # 如果未提供路径，则保持随机初始化
    if rank == 0: logger.info(f"正在加载 Stage 1 权重: {ckpt_path}")
    # 尝试以 weights_only=True 加载以保证安全性（防止反序列化漏洞）
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # 从 PyTorch Lightning 保存的 ckpt 文件中提取出纯净的 model state_dict
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    # 暴力清洗键名：移除可能因为模型嵌套导致的 'model.' 或 'net.' 前缀，确保键名能对齐
    state = {k.replace('model.', '').replace('models.', '').replace('net.', ''): v for k, v in state.items()}
    try:
        # strict=False 允许预训练权重中包含多余的层，或者当前模型包含未初始化的新层
        model.load_state_dict(state, strict=False)
    except RuntimeError:
        # 如果依然报错（通常是因为某一层存在但尺寸/通道数变了，比如分类头改变），启动智能过滤机制
        if rank == 0: logger.warning("⚠️ 检测到层结构不匹配，正在智能过滤不兼容的层...")
        curr = model.state_dict()  # 获取当前模型期待的字典结构
        # 字典推导式：只保留那些“键名存在”且“张量形状(shape)完全一致”的权重
        filtered_state = {k: v for k, v in state.items() if k in curr and v.shape == curr[k].shape}
        model.load_state_dict(filtered_state, strict=False)
    if rank == 0: logger.info("✅ 权重加载完成")


# =========================================================================
# 主程序入口
# =========================================================================

# 使用 Hydra 装饰器，自动从 ../configs 目录读取 train/mutil_train.yaml 配置文件
@hydra.main(config_path="../configs", config_name="train/mutil_train", version_base="1.3")
def main(cfg):
    # 1. 随机种子与硬件加速设置
    # 固定全局随机种子（包括 Python, NumPy, PyTorch），保证实验可完全复现
    seed_everything(cfg.seed, True)
    # 允许 RTX 30/40 系显卡使用 TF32 混合精度进行矩阵乘法，大幅加速训练但会有微小精度损失
    torch.set_float32_matmul_precision("medium")
    # 获取当前进程的编号（用于 DDP 多卡分布式训练判定）
    rank = int(os.environ.get("RANK", 0))

    # 2. 路径与实验目录配置
    # 获取数据集的名字 (例如 'imageCAS')
    d_name = list(cfg.data.keys())[0]
    # 实验名称
    run_name = f'{cfg.loss_name}'

    # 设置文档和日志保存的根目录
    base_doc_path = f"../local_results/doc/{cfg.data_name}"
    experiment_dir = os.path.join(base_doc_path, run_name)
    # 只让主进程创建文件夹，防止多进程同时创建导致系统冲突
    if rank == 0: os.makedirs(experiment_dir, exist_ok=True)

    # 3. 训练模式判定 (非常核心的设计：根据你给的后缀决定用什么 Loss)
    label_suffix = cfg.get("label_suffix", ".slice.nii.gz")
    target_loss_conf = None
    mode_name = "Unknown"

    # 根据 yaml 中的 label_suffix 自动判断当前属于哪种训练流派
    if label_suffix == ".label.nii.gz":
        mode_name = "Full Supervision (全监督真实标签)"
        target_loss_conf = cfg.loss_configs.label_loss  # 挂载全监督 Loss (Dice+CE)
    elif ".slice" in label_suffix:
        mode_name = "Slice Supervision (切片模式)"
        target_loss_conf = cfg.loss_configs.slice_loss  # 挂载稀疏切片 Loss (TV+Affinity)
    else:
        logger.warning(f"⚠️ 未知后缀 {label_suffix}，默认回退至 Label Loss")
        target_loss_conf = cfg.loss_configs.label_loss

    # 利用 Hydra 的 open_dict 解锁配置字典的只读状态，将动态选中的 Loss 注入到模块配置中
    with open_dict(cfg):
        cfg.trainer.lightning_module.loss = target_loss_conf

    if rank == 0:
        logger.info(f"📋 [训练模式判定]")
        logger.info(f"   - 标签后缀: {label_suffix}")
        logger.info(f"   - 激活模式: {mode_name}")
        logger.info(f"   - Loss类名: {target_loss_conf._target_}")

    # 设置一个进程屏障，确保所有 GPU 都判定完模式后再继续往下走
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # 4. 初始化日志记录器 (Loggers)
    loggers = [
        # Wandb: 用于将损失曲线绘制成图表 (离线模式，需后续手动同步)
        WandbLogger(save_dir=experiment_dir, name=run_name, config=OmegaConf.to_container(cfg),
                    offline=True, mode="offline"),
        # CSV: 本地保存一份纯文本的 metrics.csv
        CleanCSVLogger(save_dir=experiment_dir, name="", version="")
    ]

    # 5. 模型断点保存策略 (ModelCheckpoint)
    ckpt_cb = ModelCheckpoint(
        # 保存路径：存放到 checkpoints/imageCAS/文件夹名/实验名/ 下
        dirpath=f"{cfg.chkpt_folder}/{cfg.data_name}/{os.path.basename(os.path.normpath(cfg.data[d_name].path))}/{run_name}",
        monitor="val_DiceMetric",  # 监控验证集 Dice
        mode="max",  # Dice 越大越好，寻找最大值
        save_top_k=1,  # 只保留分数最高的 1 个权重文件
        save_last=False,  # 额外保存最后一步(last)的权重，用于防范中途崩溃
        filename="Stage2-{epoch:02d}-{val_DiceMetric:.4f}",  # 文件名格式化
        auto_insert_metric_name=False
    )
    ckpt_cb.CHECKPOINT_EQUALS_CHAR = "="
    ckpt_cb.CHECKPOINT_NAME_LAST = f"{run_name}_last"

    # 6. 数据集 (Dataset) 和 数据加载器 (DataLoader) 构建
    repeats = cfg.get("repeats", 1)  # 对同一张大图在 1 个 Epoch 里的重采样次数
    # 构建训练集：使用我们自己封装的弱监督兼容 Dataset，传入特定的后缀筛选数据
    train_dataset = MutilSupervisionDataset(
        dataset_configs=cfg.data,
        mode="train",
        finetune=True,
        repeats=repeats,
        label_suffix=label_suffix
    )
    # 构建验证集（警告：这里你使用的是 "test" 目录，这可能导致测试集泄漏，建议之后改成 "val"）
    val_dataset = UnionDataset(cfg.data, "test", finetune=True, repeats=1)
    if rank == 0:
        logger.info(f"📊 [数据集就绪]")
        logger.info(f"   - 训练集大小 (Virtual): {len(train_dataset)}")

    # 实例化 DataLoader，负责将 Dataset 的单张图片打包成 Batch，并管理多进程预读取
    train_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,  # 训练集打乱顺序
        pin_memory=True,  # 锁页内存，加快 CPU 数据移交 GPU 的速度
        drop_last=True  # 丢弃最后不够一个 Batch 的零头数据，防止 BatchNorm 计算异常
    )
    # 验证集 batch_size 强制设为 1，因为 3D 大图做滑动窗口推理极耗显存
    val_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=val_dataset, batch_size=1, shuffle=False)

    # 7. 模型初始化
    # 根据 yaml 配置实例化核心网络 (例如 DynUNet)
    base_model = hydra.utils.instantiate(cfg.model)
    # 加载预训练权重 (如果有 path_to_chkpt 的话)
    safe_load_weights(base_model, cfg.path_to_chkpt, rank)
    # 直接使用基础模型 (已经去除了之前的对比学习包装器逻辑)
    model = base_model

    # 8. 实例化 PyTorch Lightning Module
    # 它将 model(网络), loss(损失), evaluator(评估器) 封装成一个标准的计算流
    pl_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model,
        evaluator=Evaluator(),
        dataset_name=d_name
    )
    if rank == 0:
        logger.info(f"🚀 Loss 初始化完成: {type(pl_module.loss).__name__}")
        val_freq = cfg.trainer.lightning_trainer.get("val_check_interval", 1.0)
        logger.info(f"🚀 开始训练 (验证频率: 每 {val_freq} 步/Epoch)...")

    # 9. 实例化 Trainer 训练器
    trainer = hydra.utils.instantiate(
        cfg.trainer.lightning_trainer,
        logger=loggers,
        callbacks=[LearningRateMonitor(), ckpt_cb, LogCallback()],
        devices=cfg.devices,  # 取决于 yaml 中的 [0,1] 等设置
        accelerator="gpu",  # 指定使用 GPU
        # DDPStrategy 是 PyTorch 分布式训练的底层引擎。
        # find_unused_parameters=True 允许计算图中有些层没有参与当前 batch 的梯度计算而不报错
        strategy=DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=True,  # 跨多张显卡同步 BatchNorm，对医学小 batch size 尤为重要
        enable_model_summary=False,  # 关闭默认冗长的模型结构打印
    )()

    # ==========================================================
    # 10. 无缝断点续训逻辑 (Resume Training)
    # ==========================================================
    # 如果 cfg 中填写了 resume_ckpt_path，则代表我们要从崩溃/中断的地方继续
    resume_path = cfg.get("resume_ckpt_path", None)

    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            logger.info("=" * 60)
            logger.info(f"🔄 检测到断点文件，正在进行无缝续训 (Resume Training)!")
            logger.info(f"📂 路径: {resume_path}")
            logger.info("=" * 60)
        # 传入 ckpt_path，Trainer 会自动恢复：网络权重、优化器动量、学习率调度器进度、Epoch数
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=resume_path)
    else:
        # 如果没有配置断点，则代表这是一次全新的训练流程
        trainer.fit(pl_module, train_loader, val_loader)

    if rank == 0:
        logger.info("✅ 训练周期结束")


if __name__ == "__main__":
    # 劫持系统标准输出和错误输出，强行设置为行缓冲(buffering=1)
    # 这样在 Linux 服务器使用 nohup 后台运行时，日志能够实时写入文件，而不会被系统卡在缓存里
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

    # 强行关闭 wandb 在终端上的多余输出提示，保持终端清爽
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"

    # 启动主函数
    main()