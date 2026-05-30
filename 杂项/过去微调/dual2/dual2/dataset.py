import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset
from utils.mutil_supervision.mutil_dataset import MutilSupervisionDataset

logger = logging.getLogger(__name__)


class UnlabeledWeakDataset(Dataset):
    """
    纯净的无标签数据集读取器。
    它不关心标签后缀，只要是原图就读取。
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

                # 寻找原图 (排除带有 label 字样的文件)
                img_candidates = [p for p in case_dir.glob("*.nii.gz") if "label" not in p.name]
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
        import SimpleITK as sitk

        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        data_idx = torch.randint(0, len(dataset["samples"]), (1,)).item()
        sample = dataset["samples"][data_idx]

        try:
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
    """双流数据集：将 有标签Dataset 和 无标签Dataset 拼在一起"""

    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.unlabeled_len = len(unlabeled_dataset)
        self.labeled_len = len(labeled_dataset)

    def __len__(self):
        return self.unlabeled_len

    def __getitem__(self, idx):
        unlabeled_sample = self.unlabeled_dataset[idx]
        labeled_idx = torch.randint(0, self.labeled_len, (1,)).item()
        labeled_sample = self.labeled_dataset[labeled_idx]

        return {'labeled': labeled_sample, 'unlabeled': unlabeled_sample}