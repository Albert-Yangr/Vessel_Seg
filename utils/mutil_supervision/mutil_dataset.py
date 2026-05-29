import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai import transforms
from monai.transforms import Compose

logger = logging.getLogger(__name__)


# =========================================================================
# 工具函数：动态生成数据增强流水线
# =========================================================================
def generate_transforms(transforms_config: list[dict]) -> transforms.Transform:
    """
    根据 YAML 配置字典动态生成 MONAI 的数据变换管道 (Transform Pipeline)。

    原理：利用 Python 的反射机制 (getattr)，根据 YAML 里的字符串名字（如 "RandSpatialCropd"），
    直接从 monai.transforms 模块中调出对应的类，并把参数塞进去实例化。
    """
    transform_list = []
    logger.debug(f"Generating {len(transforms_config)} transforms")

    for transform_config in transforms_config:
        # 获取字典里的第一个 key，即 Transform 的名字
        transform_name = next(iter(transform_config))
        # 获取该 Transform 对应的参数字典 (kwargs)
        transform_kwargs = transform_config[transform_name]
        logger.debug(f"Generating transform {transform_name} with kwargs {transform_kwargs}")

        # 反射获取 MONAI 的 transforms 类 (相当于 eval("transforms." + transform_name))
        transform_class = getattr(transforms, transform_name)
        # 实例化该变换算子，并传入拆包后的关键字参数
        transform = transform_class(**transform_kwargs)
        transform_list.append(transform)

    # 用 Compose 将所有变换串联成一个流水线返回
    return Compose(transform_list)


# =========================================================================
# 1. 训练专用数据集：MutilSupervisionDataset
# (支持动态扫描路径、多后缀伪标签、重复重采样、按比例联合不同数据集)
# =========================================================================

class MutilSupervisionDataset(Dataset):
    def __init__(self, dataset_configs, mode, finetune=False, repeats=1, label_suffix=".pred.nii.gz"):
        """
        Args:
            dataset_configs: YAML中定义的数据集字典配置 (如 imageCAS, Parse2022 等及其采样权重)
            mode: 当前模式 ("train", "val", "test")
            finetune: 如果是微调，路径可能会去专门的子文件夹下找 (如 /path/train/)
            repeats: 数据集重复倍数。在 3D 医学分割中，由于我们使用 RandomCrop 随机裁剪，
                     为了让同一张大图在1个Epoch内被多次裁剪出不同的局部块，我们通过 repeats 虚拟放大据集长度。
            label_suffix: 目标标签后缀 (支持切换不同的伪标签，如 .slice.nii.gz, .label.nii.gz 等)
        """
        super().__init__()
        self.finetune = finetune
        self.repeats = repeats
        self.label_suffix = label_suffix

        self.datasets = []  # 存放不同数据集的元信息和路径列表
        probs = []  # 存放不同数据集的被采样概率
        self.len = 0  # 真实样本总数
        total_original = 0  # 扫描到的文件夹总数 (用于日志统计)
        total_missing = 0  # 缺少对应后缀标签的样本数 (用于日志统计)

        # 1. 遍历所有配置的数据集
        for name, config in dataset_configs.items():
            # 确定数据根目录
            data_dir = Path(config.path) / mode if finetune else Path(config.path)

            if not data_dir.exists():
                logger.warning(f"目录不存在, 已跳过: {data_dir}")
                continue

            valid_samples = []
            # 2. 遍历该数据集下的每一个病例文件夹 (如 case_001, case_002)
            for case_dir in sorted(list(data_dir.iterdir())):
                if not case_dir.is_dir():
                    continue

                total_original += 1
                case_id = case_dir.name

                # --- 查找原图 (Image) 路径 ---
                # 优先匹配含有 'img' 的文件
                img_candidates = list(case_dir.glob(f"*{case_id}*.img.nii.gz"))
                if not img_candidates:
                    # 如果没有，退而求其次：找以 .nii.gz 结尾，且不是我们要找的 label 的文件
                    img_candidates = [p for p in case_dir.glob("*.nii.gz")
                                      if self.label_suffix not in p.name and "label" not in p.name]
                if not img_candidates:
                    total_missing += 1
                    continue
                img_path = img_candidates[0]

                # --- 查找掩膜 (Mask/Label) 路径 ---
                # 寻找与我们指定的 label_suffix 完全匹配的标签文件
                label_candidates = list(case_dir.glob(f"*{self.label_suffix}"))
                if not label_candidates:
                    # 如果 glob 没搜到，尝试直接暴力拼接标准文件名进行判断
                    target_label = case_dir / f"{case_id}{self.label_suffix}"
                    if target_label.exists():
                        label_path = target_label
                    else:
                        total_missing += 1  # 找不到指定后缀的标签，作为无效样本丢弃
                        continue
                else:
                    label_path = label_candidates[0]

                # --- 数据集 ID 过滤逻辑 ---
                # 有些实验可能需要在同一个文件夹下剔除某几个特定的病例 ID
                if config.filter_dataset_IDs is not None:
                    try:
                        # 尝试从文件名解析 ID 数字 (例如 image_100.nii.gz -> 100)
                        sample_id = int(img_path.stem.split(".")[0].split("_")[-1])
                        if sample_id in config.filter_dataset_IDs:
                            continue  # 如果命中过滤名单，则跳过该样本
                    except:
                        pass

                # 将确认有效、成对的图片和标签绝对路径存入列表【核心提速点】
                valid_samples.append({
                    "img_path": str(img_path),
                    "mask_path": str(label_path)
                })

            # 如果这个数据集有有效样本，将其注册到主列表中
            if valid_samples:
                self.len += len(valid_samples)
                self.datasets.append({
                    "name": name,
                    "samples": valid_samples,
                    "transforms": generate_transforms(config.transforms[mode]),  # 生成该数据集专属的数据增强
                    "sample_prop": config.sample_prop  # 获取该数据集的采样权重
                })
                probs.append(config.sample_prop)

        # 3. 计算多数据集联合采样概率
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        # 归一化，比如权重是 [1, 2]，则概率变成 [0.33, 0.66]
        self.probs = probs_tensor / probs_tensor.sum()

        # 4. 计算虚拟总长度 = 真实图片数量 * 重复倍数
        self.virtual_len = self.len * self.repeats

        # 打印筛选结果简报
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
        # PyTorch DataLoader 会根据这个长度来决定一个 Epoch 要循环多少次 (即 Iterations = virtual_len / batch_size)
        return self.virtual_len

    def _read_nifti(self, file_path: str) -> np.ndarray:
        """底层读图方法：使用 SimpleITK 读取医学 NIfTI 图像并转化为 Numpy 数组"""
        itk_img = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(itk_img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 掷骰子：根据之前计算的概率分布 (self.probs)，随机决定当前这步抽取哪个数据集
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        retry_count = 0
        # 增加一个重试机制：防止某一张图片因为硬盘读取损坏导致整个训练崩溃
        while retry_count < 10:
            real_len = len(dataset["samples"])

            # 2. 决定抽取该数据集下的哪张图：
            if self.finetune:
                # 顺序取模策略：确保每个图片在重复时都能被均匀遍历到
                data_idx = idx % real_len
            else:
                # 完全随机策略：无视 idx，纯随机抽一张图
                data_idx = torch.randint(0, real_len, (1,)).item()

            sample = dataset["samples"][data_idx]

            try:
                # 3. 读取图像并转为单精度浮点数 (float32)，符合模型输入要求
                img = self._read_nifti(sample['img_path']).astype(np.float32)
                # 4. 动态解析掩膜 (Mask) 的数据类型
                if ".sdf" in self.label_suffix.lower():
                    # 💡 注意：SDF (符号距离场) 标签里面包含了距离边界的实际物理距离，有负数和小数，必须用 float32
                    mask = self._read_nifti(sample['mask_path']).astype(np.float32)
                else:
                    # 常规的类别标签 (0背景, 1血管, 255忽略) 使用无符号8位整数 (uint8) 就够了，大幅节省内存
                    mask = self._read_nifti(sample['mask_path']).astype(np.uint8)
                # 5. 应用 MONAI 数据增强管道 (如裁剪、旋转、加噪等)
                # 传入字典，它会同时、同步地对 Image 和 Mask 施加相同的几何变换
                transformed = dataset['transforms']({'Image': img, 'Mask': mask})
                # MONAI 的某些 Crop 算子（如 RandCropByPosNegLabeld）可能会返回包含多个块的列表。
                # 由于我们在 config 里设置了 batch 维度组合，这里如果返回列表则只取第一个元素
                if isinstance(transformed, list):
                    transformed = transformed[0]

                return transformed['Image'], transformed['Mask']

            except Exception as e:
                # 如果遇到错误（例如文件损坏），随机换一个新的 idx 重新尝试
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

        # 如果连续 10 次抽取都崩溃，说明数据集大概率整体损坏了，抛出异常
        raise RuntimeError("DataLoader 连续重试多次失败，请检查硬盘数据是否损坏！")


# =========================================================================
# 2. 验证集/测试集专用：UnionDataset
# (设计哲学：稳定、不使用伪标签、不需要复杂的后缀匹配、直接固定读取 GT)
# =========================================================================

class UnionDataset(Dataset):
    def __init__(self, dataset_configs, mode, finetune=False, repeats=1):
        super().__init__()
        self.finetune = finetune
        self.repeats = repeats
        self.datasets, probs = [], []
        self.len = 0

        # 初始化扫描逻辑与训练集类似，但更简单，不关心后缀，只关心文件夹
        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)

            if not data_dir.exists():
                logger.warning(f"验证/测试目录不存在: {data_dir}")
                continue

            # 验证集强制要求排序，以保证每次跑验证/推理时，切片的顺序是一致的
            paths = sorted(list(data_dir.iterdir()))

            self.len += len(paths)
            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,  # 只存文件夹路径
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

    def _read_nifti(self, file_path: str) -> np.ndarray:
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
                # 验证集数据一定包含最规整的金标准 (Ground Truth)
                # 直接在目录下模糊匹配名字带 'img' 和 'label' 的文件
                img_path = [p for p in sample_id.iterdir() if 'img' in p.name][0]
                mask_path = [p for p in sample_id.iterdir() if 'label' in p.name][0]
            except IndexError:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

            # ID 过滤逻辑
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
                # 读取图像，依然转为 float32
                img = self._read_nifti(img_path).astype(np.float32)
                # 💡 验证集的掩膜强制转为 bool (布尔型：True/False)。
                # 这是因为验证集不需要理会 255 等忽略标签，只关心严格的二值对错，用于计算 Dice 等硬指标。
                mask = self._read_nifti(mask_path).astype(bool)

                transformed = dataset['transforms']({'Image': img, 'Mask': mask})

                if isinstance(transformed, list):
                    transformed = transformed[0]

                # 💡 再次确保返回的 Mask 是二值的张量格式 (大于0的变为True/1)
                return transformed['Image'], transformed['Mask'] > 0

            except Exception as e:
                idx = torch.randint(0, self.virtual_len, (1,)).item()
                retry_count += 1
                continue

        raise RuntimeError("Failed to load valid data after multiple retries.")