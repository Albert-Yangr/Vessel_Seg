import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch
from monai import transforms
from monai.transforms import Compose
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# 数据文件总览
# -----------------------------------------------------------------------------
# 当前基础 dual_train 使用“双流训练 + 单流验证”的组织方式：
#
# 训练流：
#   MutilSupervisionDataset  -> 读取有标签/弱监督切片标签数据。
#   UnlabeledWeakDataset     -> 读取无标签原图。
#   DualStreamDataset        -> 将两者配对成 {"labeled": ..., "unlabeled": ...}。
#
# 验证流：
#   UnionDataset             -> 读取完整验证/测试图像和完整 label，用于 Dice。
# =============================================================================


def generate_transforms(transforms_config: list[dict]) -> transforms.Transform:
    """
    根据 yaml 中配置的 MONAI transforms 动态构建 Compose。

    transforms_config 形如：
        - LoadImaged: {...}
        - RandCropByPosNegLabeld: {...}

    这里会按顺序实例化对应 MONAI transform，并组合成 Compose。
    """
    transform_list = []
    for transform_config in transforms_config:
        transform_name = next(iter(transform_config))
        transform_kwargs = transform_config[transform_name]
        transform_class = getattr(transforms, transform_name)
        transform_list.append(transform_class(**transform_kwargs))
    return Compose(transform_list)


class UnlabeledWeakDataset(Dataset):
    """
    无标签训练数据集。

    作用：
        只读取无标签原图，不读取真实标签，用于 CPS 中的伪标签损失。

    为什么返回 dummy Mask：
        训练增强通常复用 MONAI 的 {"Image", "Mask"} 字典格式。无标签数据没有
        真实 Mask，因此这里创建全 0 dummy_mask 只为通过 transform 流程。
        后续 module.py 中会执行 img_u, _ = unpack(batch_u)，这个 Mask 会被丢弃。

    长度：
        virtual_len = 原始无标签病例数 * repeats。
    """

    def __init__(self, dataset_configs, mode, finetune=False, repeats=1):
        self.repeats = repeats
        self.datasets = []
        self.len = 0

        for name, config in dataset_configs.items():
            data_dir = Path(config.path)
            if not data_dir.exists():
                logger.warning(f"Unlabeled data dir not found, skipped: {data_dir}")
                continue

            valid_samples = []
            for case_dir in sorted(data_dir.iterdir()):
                if not case_dir.is_dir():
                    continue

                # 优先找标准命名的原图；找不到时，退化为排除 label/slice 的 nii.gz。
                img_candidates = list(case_dir.glob("*img.nii.gz"))
                if not img_candidates:
                    img_candidates = [
                        p for p in case_dir.glob("*.nii.gz")
                        if "label" not in p.name and "slice" not in p.name
                    ]
                if not img_candidates:
                    continue

                valid_samples.append({"img_path": str(img_candidates[0])})

            if valid_samples:
                self.datasets.append(
                    {
                        "name": name,
                        "samples": valid_samples,
                        "transforms": generate_transforms(config.transforms[mode]),
                        "sample_prop": config.sample_prop,
                    }
                )
                self.len += len(valid_samples)

        if self.len <= 0:
            raise RuntimeError("Unlabeled dataset is empty. Please check train-all path.")

        probs = torch.tensor([d["sample_prop"] for d in self.datasets], dtype=torch.float32)
        self.probs = probs / probs.sum()
        self.virtual_len = self.len * self.repeats

        logger.info("=" * 50)
        logger.info(f"[UnlabeledWeakDataset] loaded unlabeled images: {self.len}")
        logger.info(f"[UnlabeledWeakDataset] virtual length x{self.repeats}: {self.virtual_len}")
        logger.info("=" * 50)

    def __len__(self):
        return self.virtual_len

    @staticmethod
    def _read_niigz(file_path: str) -> np.ndarray:
        itk_img = sitk.ReadImage(str(file_path))
        return sitk.GetArrayFromImage(itk_img)

    def __getitem__(self, idx: int):
        # 多数据集混合时，按 sample_prop 随机选择数据集。
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        # 无标签流采用随机病例采样；idx 只决定 epoch 长度，不强制顺序读取。
        data_idx = torch.randint(0, len(dataset["samples"]), (1,)).item()
        sample = dataset["samples"][data_idx]

        try:
            img = self._read_niigz(sample["img_path"]).astype(np.float32)
            dummy_mask = np.zeros_like(img, dtype=np.uint8)

            transformed = dataset["transforms"]({"Image": img, "Mask": dummy_mask})
            if isinstance(transformed, list):
                transformed = transformed[0]
            return {"Image": transformed["Image"], "Mask": transformed["Mask"]}
        except Exception:
            # 数据损坏时随机换一个样本，避免 DataLoader 直接中断。
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())


class MutilSupervisionDataset(Dataset):
    """
    有标签/弱监督训练数据集。

    作用：
        读取有标签训练样本，当前基础实验中通常读取 `.slice.nii.gz` 切片弱监督标签。

    路径：
        dual_train.py 会把 labeled path 指向 `.../train`。

    返回：
        (Image, Mask)

    用途：
        作为 DualStreamDataset 的 labeled_dataset，用于监督损失 loss_sup。
    """

    def __init__(self, dataset_configs, mode, finetune=False, repeats=1, label_suffix=".pred.nii.gz"):
        super().__init__()
        self.finetune = finetune
        self.repeats = repeats
        self.label_suffix = label_suffix
        self.datasets = []
        self.len = 0

        probs = []
        total_original = 0
        total_missing = 0

        for name, config in dataset_configs.items():
            data_dir = Path(config.path) / mode if finetune else Path(config.path)
            if not data_dir.exists():
                logger.warning(f"Labeled data dir not found, skipped: {data_dir}")
                continue

            valid_samples = []
            for case_dir in sorted(data_dir.iterdir()):
                if not case_dir.is_dir():
                    continue

                total_original += 1
                case_id = case_dir.name

                img_candidates = list(case_dir.glob(f"*{case_id}*.img.nii.gz"))
                if not img_candidates:
                    img_candidates = [
                        p for p in case_dir.glob("*.nii.gz")
                        if self.label_suffix not in p.name and "label" not in p.name
                    ]
                if not img_candidates:
                    total_missing += 1
                    continue
                img_path = img_candidates[0]

                label_candidates = list(case_dir.glob(f"*{self.label_suffix}"))
                if label_candidates:
                    label_path = label_candidates[0]
                else:
                    target_label = case_dir / f"{case_id}{self.label_suffix}"
                    if not target_label.exists():
                        total_missing += 1
                        continue
                    label_path = target_label

                if config.filter_dataset_IDs is not None:
                    try:
                        sample_id = int(img_path.name.split(".")[0].split("_")[-1])
                        if sample_id in config.filter_dataset_IDs:
                            continue
                    except Exception:
                        pass

                valid_samples.append({"img_path": str(img_path), "mask_path": str(label_path)})

            if valid_samples:
                self.datasets.append(
                    {
                        "name": name,
                        "samples": valid_samples,
                        "transforms": generate_transforms(config.transforms[mode]),
                        "sample_prop": config.sample_prop,
                    }
                )
                self.len += len(valid_samples)
                probs.append(config.sample_prop)

        if self.len <= 0:
            raise RuntimeError(f"No labeled samples found with suffix: {self.label_suffix}")

        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        self.probs = probs_tensor / probs_tensor.sum()
        self.virtual_len = self.len * self.repeats

        logger.info("=" * 50)
        logger.info(f"[MutilSupervisionDataset] label suffix: {self.label_suffix}")
        logger.info(f"[MutilSupervisionDataset] scanned cases: {total_original}")
        logger.info(f"[MutilSupervisionDataset] missing/invalid: {total_missing}")
        logger.info(f"[MutilSupervisionDataset] valid labeled samples: {self.len}")
        logger.info(f"[MutilSupervisionDataset] virtual length x{self.repeats}: {self.virtual_len}")
        logger.info("=" * 50)

    def __len__(self):
        return self.virtual_len

    @staticmethod
    def _read_niigz(file_path: str) -> np.ndarray:
        itk_img = sitk.ReadImage(str(file_path))
        return sitk.GetArrayFromImage(itk_img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        retry_count = 0
        while retry_count < 10:
            real_len = len(dataset["samples"])
            # finetune/validation 风格使用顺序取样；训练风格使用随机取样。
            data_idx = idx % real_len if self.finetune else torch.randint(0, real_len, (1,)).item()
            sample = dataset["samples"][data_idx]

            try:
                img = self._read_niigz(sample["img_path"]).astype(np.float32)
                if ".sdf" in self.label_suffix.lower():
                    mask = self._read_niigz(sample["mask_path"]).astype(np.float32)
                else:
                    mask = self._read_niigz(sample["mask_path"]).astype(np.uint8)

                transformed = dataset["transforms"]({"Image": img, "Mask": mask})
                if isinstance(transformed, list):
                    transformed = transformed[0]
                return transformed["Image"], transformed["Mask"]
            except Exception:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1

        raise RuntimeError("Failed to load labeled data after multiple retries.")


class DualStreamDataset(Dataset):
    """
    双流配对数据集。

    作用：
        将一个有标签样本和一个无标签样本打包成一个训练样本：

            {
                "labeled": labeled_sample,
                "unlabeled": unlabeled_sample
            }

    长度策略：
        由无标签流决定 epoch 长度。这样可以让无标签数据得到充分利用。

    有标签采样：
        cycle:  idx % labeled_len，循环使用有标签样本。
        random: 每次随机抽一个有标签样本。
    """

    def __init__(self, labeled_dataset, unlabeled_dataset, labeled_sampling="cycle"):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.unlabeled_len = len(unlabeled_dataset)
        self.labeled_len = len(labeled_dataset)
        self.labeled_sampling = labeled_sampling

        logger.info("=" * 50)
        logger.info("[DualStreamDataset] paired training stream")
        logger.info(f"  unlabeled virtual length: {self.unlabeled_len}")
        logger.info(f"  labeled virtual length: {self.labeled_len}")
        logger.info(f"  epoch length: {self.unlabeled_len}")
        logger.info(f"  labeled sampling: {self.labeled_sampling}")
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
        return {"labeled": labeled_sample, "unlabeled": unlabeled_sample}


class UnionDataset(Dataset):
    """
    验证/测试数据集。

    作用：
        读取完整图像和完整 label，用于 validation_step 中的滑窗推理和 Dice 计算。

    返回：
        (Image, Mask > 0)
    """

    def __init__(self, dataset_configs, mode, finetune=False, repeats=1):
        super().__init__()
        self.finetune = finetune
        self.repeats = repeats
        self.datasets = []
        self.len = 0
        probs = []

        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
            if not data_dir.exists():
                logger.warning(f"Validation/test data dir not found, skipped: {data_dir}")
                continue

            paths = sorted([p for p in data_dir.iterdir() if p.is_dir()])
            self.len += len(paths)
            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                    "sample_prop": dataset_config.sample_prop,
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs,
                }
            )
            probs.append(dataset_config.sample_prop)

        if self.len <= 0:
            raise RuntimeError("Validation/test dataset is empty.")

        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        self.probs = probs_tensor / probs_tensor.sum()
        self.virtual_len = self.len * self.repeats

    def __len__(self):
        return self.virtual_len

    @staticmethod
    def _read_niigz(file_path: str) -> np.ndarray:
        itk_img = sitk.ReadImage(str(file_path))
        return sitk.GetArrayFromImage(itk_img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        retry_count = 0
        while retry_count < 10:
            data_idx = idx % len(dataset["paths"]) if self.finetune else torch.randint(0, len(dataset["paths"]), (1,)).item()
            case_dir = dataset["paths"][data_idx]

            try:
                img_path = [p for p in case_dir.iterdir() if "img" in p.name and p.name.endswith(".nii.gz")][0]
                mask_path = [p for p in case_dir.iterdir() if "label" in p.name and p.name.endswith(".nii.gz")][0]
            except IndexError:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

            if dataset["filter_dataset_IDs"] is not None:
                try:
                    sample_dataset_id = int(img_path.name.split(".")[0].split("_")[-1])
                    if sample_dataset_id in dataset["filter_dataset_IDs"]:
                        idx = torch.randint(0, self.virtual_len, (1,)).item()
                        retry_count += 1
                        continue
                except Exception:
                    pass

            try:
                img = self._read_niigz(img_path).astype(np.float32)
                mask = self._read_niigz(mask_path).astype(bool)
                transformed = dataset["transforms"]({"Image": img, "Mask": mask})
                if isinstance(transformed, list):
                    transformed = transformed[0]
                return transformed["Image"], transformed["Mask"] > 0
            except Exception:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1

        raise RuntimeError("Failed to load validation/test data after multiple retries.")
