import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset
from utils.mutil_supervision.mutil_dataset import MutilSupervisionDataset
import SimpleITK as sitk  # 🌟 新增导入

logger = logging.getLogger(__name__)


class UnlabeledWeakDataset(Dataset):
    """
    纯净的无标签数据集读取器。
    它不关心标签后缀，只要是原图就读取 (现已适配 .nii.gz 格式)。
    """

    def __init__(self, dataset_configs, mode, finetune=False, repeats=1):
        self.repeats = repeats
        self.datasets = []
        self.len = 0

        from utils.full_train_dataset import generate_transforms

        for name, config in dataset_configs.items():
            data_dir = Path(config.path)
            if not data_dir.exists():
                continue

            valid_samples = []
            for case_dir in sorted(list(data_dir.iterdir())):
                if not case_dir.is_dir():
                    continue

                # 🌟 修改点 1：寻找原图 .nii.gz (排除带有 label 和 slice 字样的文件)
                img_candidates = list(case_dir.glob("*img.nii.gz"))
                if not img_candidates:
                    img_candidates = [p for p in case_dir.glob("*.nii.gz") if
                                      "label" not in p.name and "slice" not in p.name]

                if not img_candidates:
                    continue

                valid_samples.append({
                    "img_path": str(img_candidates[0]),
                })

            if valid_samples:
                self.datasets.append({
                    "name": name,
                    "samples": valid_samples,
                    "transforms": generate_transforms(config.transforms[mode]),
                    "sample_prop": config.sample_prop
                })

        # 计算采样概率和总长度
        self.len = sum(len(d["samples"]) for d in self.datasets)
        self.virtual_len = self.len * self.repeats

        if self.len > 0:
            probs = [d["sample_prop"] for d in self.datasets]
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            self.probs = probs_tensor / probs_tensor.sum()

            logger.info("=" * 50)
            logger.info(f"🟢 [无标签通道] 成功扫描载入无标签原图: {self.len} 张！")
            logger.info("=" * 50)
        else:
            raise RuntimeError("⚠️ 无标签数据集为空，请检查路径！")

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx: int):
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        data_idx = torch.randint(0, len(dataset["samples"]), (1,)).item()
        sample = dataset["samples"][data_idx]

        try:
            # 🌟 修改点 2：换回 SimpleITK 读取医学图像
            itk_img = sitk.ReadImage(sample['img_path'])
            img = sitk.GetArrayFromImage(itk_img).astype(np.float32)

            # 🌟 这里是全 0 假 Mask。
            # 作用仅仅是为了满足 MONAI Transforms 必须同时输入 Image 和 Mask 的 API 格式要求！
            # 它在 Module 中会被彻底丢弃，绝不会参与 Loss 计算和伪标签生成！
            dummy_mask = np.zeros_like(img, dtype=np.uint8)

            data_dict = {'Image': img, 'Mask': dummy_mask}
            transformed = dataset['transforms'](data_dict)
            if isinstance(transformed, list):
                transformed = transformed[0]

            return {'Image': transformed['Image'], 'Mask': transformed['Mask']}
        except Exception as e:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())


class DualStreamDataset(Dataset):
    """
    双流数据集：每个无标签样本配一个有标签样本。

    一个 epoch 的长度由无标签流决定，例如 85 个无标签病例 repeats=10 后就是 850。
    有标签流不会被实体复制到 850 份，而是在这 850 个 step 中循环取模采样；
    如果底层有标签 Dataset 本身做随机增强，同一个病例会自然产生不同 crop。
    """

    def __init__(self, labeled_dataset, unlabeled_dataset, labeled_sampling="cycle"):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.unlabeled_len = len(unlabeled_dataset)
        self.labeled_len = len(labeled_dataset)
        self.labeled_sampling = labeled_sampling

        logger.info("=" * 50)
        logger.info("🔗 [双流配对策略]")
        logger.info(f"   - 无标签虚拟长度: {self.unlabeled_len}")
        logger.info(f"   - 有标签虚拟长度: {self.labeled_len}")
        logger.info(f"   - Epoch 长度采用无标签流: {self.unlabeled_len}")
        logger.info(f"   - 有标签采样方式: {self.labeled_sampling}")
        logger.info("=" * 50)

    def __len__(self):
        return self.unlabeled_len

    def __getitem__(self, idx):
        unlabeled_sample = self.unlabeled_dataset[idx]
        if self.labeled_sampling == "random":
            labeled_idx = torch.randint(0, self.labeled_len, (1,)).item()
        else:
            labeled_idx = idx % self.labeled_len
        labeled_sample = self.labeled_dataset[labeled_idx]

        return {'labeled': labeled_sample, 'unlabeled': unlabeled_sample}


import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms
from monai.transforms import Compose
import SimpleITK as sitk  # 🌟 新增导入

logger = logging.getLogger(__name__)


# =========================================================================
# 工具函数：动态生成数据增强流水线
# =========================================================================
def generate_transforms(transforms_config: list[dict]) -> transforms.Transform:
    transform_list = []
    logger.debug(f"Generating {len(transforms_config)} transforms")

    for transform_config in transforms_config:
        transform_name = next(iter(transform_config))
        transform_kwargs = transform_config[transform_name]
        logger.debug(f"Generating transform {transform_name} with kwargs {transform_kwargs}")

        transform_class = getattr(transforms, transform_name)
        transform = transform_class(**transform_kwargs)
        transform_list.append(transform)

    return Compose(transform_list)


# =========================================================================
# 1. 训练专用数据集：MutilSupervisionDataset
# =========================================================================

class MutilSupervisionDataset(Dataset):
    def __init__(self, dataset_configs, mode, finetune=False, repeats=1, label_suffix=".pred.nii.gz"):
        super().__init__()
        self.finetune = finetune
        self.repeats = repeats
        self.label_suffix = label_suffix

        self.datasets = []
        probs = []
        self.len = 0
        total_original = 0
        total_missing = 0

        for name, config in dataset_configs.items():
            data_dir = Path(config.path) / mode if finetune else Path(config.path)

            if not data_dir.exists():
                logger.warning(f"目录不存在, 已跳过: {data_dir}")
                continue

            valid_samples = []
            for case_dir in sorted(list(data_dir.iterdir())):
                if not case_dir.is_dir():
                    continue

                total_original += 1
                case_id = case_dir.name

                # --- 🌟 查找原图 (Image) 路径，修改为查找 nii.gz ---
                img_candidates = list(case_dir.glob(f"*{case_id}*.img.nii.gz"))
                if not img_candidates:
                    img_candidates = [p for p in case_dir.glob("*.nii.gz")
                                      if self.label_suffix not in p.name and "label" not in p.name]
                if not img_candidates:
                    total_missing += 1
                    continue
                img_path = img_candidates[0]

                # --- 查找掩膜 (Mask/Label) 路径 ---
                label_candidates = list(case_dir.glob(f"*{self.label_suffix}"))
                if not label_candidates:
                    target_label = case_dir / f"{case_id}{self.label_suffix}"
                    if target_label.exists():
                        label_path = target_label
                    else:
                        total_missing += 1
                        continue
                else:
                    label_path = label_candidates[0]

                # --- 数据集 ID 过滤逻辑 ---
                if config.filter_dataset_IDs is not None:
                    try:
                        # 尝试从文件名解析 ID (支持类似于 .nii.gz 的双重后缀)
                        sample_id = int(img_path.name.split(".")[0].split("_")[-1])
                        if sample_id in config.filter_dataset_IDs:
                            continue
                    except:
                        pass

                valid_samples.append({
                    "img_path": str(img_path),
                    "mask_path": str(label_path)
                })

            if valid_samples:
                self.len += len(valid_samples)
                self.datasets.append({
                    "name": name,
                    "samples": valid_samples,
                    "transforms": generate_transforms(config.transforms[mode]),
                    "sample_prop": config.sample_prop
                })
                probs.append(config.sample_prop)

        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        self.probs = probs_tensor / probs_tensor.sum()
        self.virtual_len = self.len * self.repeats

        logger.info("=" * 50)
        logger.info(f"🔍 [弱监督数据筛选报告] Suffix: {self.label_suffix}")
        logger.info(f"   - 原始扫描总数: {total_original}")
        logger.info(f"   - ❌ 无效/无标签: {total_missing}")
        logger.info(f"   - ✅ 有效样本数: {self.len}")
        logger.info(f"   - 📚 实际训练长度 (x{self.repeats}): {self.virtual_len}")
        logger.info("=" * 50)

        if self.virtual_len == 0:
            raise RuntimeError(f"❌ 错误: 未找到带有 {self.label_suffix} 的数据！请检查数据路径。")

    def __len__(self):
        return self.virtual_len

    # 🌟 修改底层读图方法
    def _read_niigz(self, file_path: str) -> np.ndarray:
        """底层读图方法：使用 SimpleITK 读取医学图像"""
        itk_img = sitk.ReadImage(str(file_path))
        return sitk.GetArrayFromImage(itk_img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        retry_count = 0
        while retry_count < 10:
            real_len = len(dataset["samples"])

            if self.finetune:
                data_idx = idx % real_len
            else:
                data_idx = torch.randint(0, real_len, (1,)).item()

            sample = dataset["samples"][data_idx]

            try:
                # 🌟 替换为 _read_niigz
                img = self._read_niigz(sample['img_path']).astype(np.float32)

                if ".sdf" in self.label_suffix.lower():
                    mask = self._read_niigz(sample['mask_path']).astype(np.float32)
                else:
                    mask = self._read_niigz(sample['mask_path']).astype(np.uint8)

                transformed = dataset['transforms']({'Image': img, 'Mask': mask})
                if isinstance(transformed, list):
                    transformed = transformed[0]

                return transformed['Image'], transformed['Mask']

            except Exception as e:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

        raise RuntimeError("DataLoader 连续重试多次失败，请检查硬盘数据是否损坏！")


# =========================================================================
# 2. 验证集/测试集专用：UnionDataset
# =========================================================================

class UnionDataset(Dataset):
    def __init__(self, dataset_configs, mode, finetune=False, repeats=1):
        super().__init__()
        self.finetune = finetune
        self.repeats = repeats
        self.datasets, probs = [], []
        self.len = 0

        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)

            if not data_dir.exists():
                logger.warning(f"验证/测试目录不存在: {data_dir}")
                continue

            paths = sorted(list(data_dir.iterdir()))

            self.len += len(paths)
            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                    "sample_prop": dataset_config.sample_prop,
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs
                }
            )
            probs.append(dataset_config.sample_prop)

        probs = torch.tensor(probs, dtype=torch.float32)
        self.probs = probs / probs.sum()
        self.virtual_len = self.len * self.repeats

    def __len__(self):
        return self.virtual_len

    # 🌟 修改底层读图方法
    def _read_niigz(self, file_path: str) -> np.ndarray:
        itk_img = sitk.ReadImage(str(file_path))
        return sitk.GetArrayFromImage(itk_img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        retry_count = 0
        while retry_count < 10:
            if self.finetune:
                real_len = len(dataset["paths"])
                data_idx = idx % real_len
            else:
                data_idx = torch.randint(0, len(dataset["paths"]), (1,)).item()

            sample_id = dataset["paths"][data_idx]

            try:
                # 🌟 强制增加 .nii.gz 后缀判定
                img_path = [p for p in sample_id.iterdir() if 'img' in p.name and p.name.endswith('.nii.gz')][0]
                mask_path = [p for p in sample_id.iterdir() if 'label' in p.name and p.name.endswith('.nii.gz')][0]
            except IndexError:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

            if dataset['filter_dataset_IDs'] is not None:
                try:
                    sample_dataset_id = int(img_path.name.split(".")[0].split("_")[-1])
                    if sample_dataset_id in dataset['filter_dataset_IDs']:
                        idx = torch.randint(0, self.virtual_len, (1,)).item()
                        retry_count += 1
                        continue
                except:
                    pass

            try:
                # 🌟 替换为 _read_niigz
                img = self._read_niigz(img_path).astype(np.float32)
                mask = self._read_niigz(mask_path).astype(bool)

                transformed = dataset['transforms']({'Image': img, 'Mask': mask})

                if isinstance(transformed, list):
                    transformed = transformed[0]

                return transformed['Image'], transformed['Mask'] > 0

            except Exception as e:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

        raise RuntimeError("Failed to load valid data after multiple retries.")
