import logging
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms
from monai.transforms import Compose

from utils.io import determine_reader_writer

logger = logging.getLogger(__name__)


def generate_transforms(
        transforms_config: list[dict],
) -> list[transforms.Transform]:
    """根据配置生成数据变换管道"""
    transform_list = []
    logger.debug(f"Generating {len(transforms_config)} transforms")

    for transform_config in transforms_config:
        transform_name = next(iter(transform_config))
        transform_kwargs = transform_config[transform_name]
        logger.debug(
            f"Generating transform {transform_name} with kwargs {transform_kwargs}"
        )
        transform: transforms.Transform = getattr(transforms, transform_name)(
            **transform_kwargs
        )  # type: ignore
        transform_list.append(transform)

    return Compose(transform_list)  # type: ignore


class UnionDataset(Dataset):
    """
    联合数据集类 (支持重复采样 repeats)
    """

    def __init__(self, dataset_configs, mode, finetune=False, repeats=1):
        """
        Args:
            repeats (int): 数据集重复次数。如果设为 8，100个样本会被视为 800 个。
                           配合 RandomCrop，相当于每个 epoch 从每张图取 8 个不同的块。
        """
        super().__init__()
        self.finetune = finetune
        self.repeats = repeats  # 新增重复次数
        self.datasets, probs = [], []
        self.len = 0

        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
            paths = sorted(list(data_dir.iterdir()))

            self.len += len(paths)
            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,
                    "reader": determine_reader_writer(dataset_config.file_format)(),
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                    "sample_prop": dataset_config.sample_prop,
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs
                }
            )
            probs.append(dataset_config.sample_prop)

        probs = torch.tensor(probs)
        self.probs = probs / probs.sum()

        # 核心逻辑：总长度乘以重复次数
        self.virtual_len = self.len * self.repeats

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 采样数据集
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        retry_count = 0
        while retry_count < 10:
            # 2. 确定样本索引
            if self.finetune:
                # 这里的 idx 范围是 0 ~ (100 * repeats - 1)
                # 取模操作确保映射回 0 ~ 99 的真实文件索引
                real_len = len(dataset["paths"])
                data_idx = idx % real_len
            else:
                data_idx = torch.randint(0, len(dataset["paths"]), (1,)).item()

            sample_id = dataset["paths"][data_idx]

            # 3. 查找文件
            try:
                img_path = [p for p in sample_id.iterdir() if 'img' in p.name][0]
                mask_path = [p for p in sample_id.iterdir() if 'label' in p.name][0]
            except IndexError:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

            # 4. 过滤逻辑
            if dataset['filter_dataset_IDs'] is not None:
                try:
                    sample_dataset_id = int(img_path.stem.split("_")[-1])
                    if sample_dataset_id in dataset['filter_dataset_IDs']:
                        idx = torch.randint(0, self.virtual_len, (1,)).item()
                        retry_count += 1
                        continue
                except:
                    pass

            try:
                # 5. 读取与变换
                img = dataset['reader'].read_images(str(img_path))[0].astype(np.float32)
                mask = dataset['reader'].read_images(str(mask_path))[0].astype(bool)

                transformed = dataset['transforms']({'Image': img, 'Mask': mask})

                # --- 【核心修复】 ---
                # 处理 RandCrop 可能返回列表的情况 (即便是 num_samples=1 也可能是 list)
                if isinstance(transformed, list):
                    transformed = transformed[0]
                # ------------------

                return transformed['Image'], transformed['Mask'] > 0

            except Exception as e:
                # print(f"Warning: Error loading {sample_id}: {e}")
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

        raise RuntimeError("Failed to load valid data after multiple retries.")