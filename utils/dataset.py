import logging
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.io import determine_reader_writer

logger = logging.getLogger(__name__)


# ========================================================

import logging

from monai import transforms
from monai.transforms import Compose

logger = logging.getLogger(__name__)


def generate_transforms(
        transforms_config: list[dict],
) -> list[transforms.Transform]:
    """
    根据配置生成数据变换管道
    训练和推理过程中使用的数据增强和预处理变换。
    Args:
        transforms_config (list[dict]): 变换配置列表，每个字典包含变换名称和参数
    Returns:
        Compose: 组合后的变换管道对象
    """

    transform_list = []  # 存储变换对象的列表
    logger.debug(f"Generating {len(transforms_config)} transforms")  # 记录变换数量

    # 遍历所有变换配置
    for transform_config in transforms_config:
        # 获取变换名称（字典的第一个键）
        transform_name = next(iter(transform_config))
        # 获取变换的参数配置
        transform_kwargs = transform_config[transform_name]
        logger.debug(
            f"Generating transform {transform_name} with kwargs {transform_kwargs}"
        )
        # 使用反射机制动态创建变换对象
        # 从MONAI的transforms模块中获取对应的变换类并实例化
        transform: transforms.Transform = getattr(transforms, transform_name)(
            **transform_kwargs  # 使用关键字参数初始化
        )  # type: ignore
        transform_list.append(transform)  # 将变换添加到列表中

    # 将所有变换组合成一个管道，按顺序执行
    return Compose(transform_list)  # type: ignore

# ========================================================




class UnionDataset(Dataset):
    """
    联合数据集类，用于将多个数据集合并为一个统一的数据集。
    支持在训练时按比例采样不同数据集，以及在微调时使用固定样本。
    """

    def __init__(self, dataset_configs, mode, finetune=False):
        """
        初始化联合数据集

        Args:
            dataset_configs: 数据集配置字典，包含各个数据集的配置信息
            mode: 数据集模式（如'fine-tuning', 'val', 'test'）
            finetune: 是否为微调模式，影响数据采样策略
        """
        super().__init__()
        # 初始化数据集
        self.finetune = finetune
        self.datasets, probs = [], []  # 存储数据集信息和采样概率
        self.len = 0  # 数据集总长度

        # 遍历所有数据集配置
        for name, dataset_config in dataset_configs.items():
            # 构建数据目录路径：微调模式下使用mode子目录，否则使用根目录
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
            # 排序确保在微调模式下使用相同的样本（特别是1-shot场景）
            paths = sorted(list(data_dir.iterdir()))

            # 累加数据集长度
            self.len += len(paths)
            # 存储数据集信息
            self.datasets.append(
                {
                    "name": name,  # 数据集名称
                    "paths": paths,  # 数据样本路径列表
                    "reader": determine_reader_writer(dataset_config.file_format)(),  # 数据读取器
                    "transforms": generate_transforms(dataset_config.transforms[mode]),  # 数据变换
                    "sample_prop": dataset_config.sample_prop,  # 采样比例
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs  # 过滤的样本ID
                }
            )
            probs.append(dataset_config.sample_prop)  # 收集采样概率

        # 确保概率总和为1（归一化处理）
        probs = torch.tensor(probs)
        self.probs = probs / probs.sum()  # 归一化后的采样概率

    def __len__(self):
        """返回数据集总长度"""
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个数据样本
        Args:
            idx: 样本索引
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 图像和对应的分割掩码
        """
        # 第一步：根据概率分布采样数据集
        dataset_id = torch.multinomial(self.probs, 1).item()  # 多项式采样
        dataset = self.datasets[dataset_id]
        # 第二步：采样数据样本（使用循环确保满足过滤条件）
        while True:
            # 微调模式下使用固定索引，训练模式下随机采样
            data_idx = idx if self.finetune else torch.randint(0, len(dataset["paths"]), (1,)).item()
            sample_id = dataset["paths"][data_idx]  # 获取样本目录

            # 查找图像和掩码文件（假设文件名包含'img'和'mask'）
            img_path = [path for path in sample_id.iterdir() if 'img' in path.name][0]
            mask_path = [path for path in sample_id.iterdir() if 'label' in path.name][0]

            # 检查是否需要过滤该样本（基于数据集ID）
            if dataset['filter_dataset_IDs'] is not None:
                # 从文件名中提取数据集ID（假设格式为xxx_ID）
                sample_dataset_id = int(img_path.stem.split("_")[-1])
                if sample_dataset_id in dataset['filter_dataset_IDs']:
                    continue  # 跳过被过滤的样本

            # 读取图像和掩码数据
            img = dataset['reader'].read_images(str(img_path))[0].astype(np.float32)
            mask = dataset['reader'].read_images(str(mask_path))[0].astype(bool)

            # 应用数据增强/预处理变换
            transformed = dataset['transforms']({'Image': img, 'Mask': mask})

            # 返回处理后的图像和二值化的掩码
            return transformed['Image'], transformed['Mask'] > 0



