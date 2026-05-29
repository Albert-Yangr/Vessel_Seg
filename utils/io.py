"""Reader/writer classes; we follow https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/imageio"""
"""
医学图像读写器类 - 支持多种医学图像格式

基于nnUNet的图像IO模块设计，为vesselFM提供统一的数据接口
支持论文中提到的多种成像模态数据格式
"""

import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class BaseReaderWriter(ABC):
    """基础读写器抽象基类

    定义统一的医学图像读写接口，确保不同格式数据的一致性处理
    对应论文中处理多种成像模态数据的需求
    """
    supported_file_formats = []  # 支持的文件格式列表

    @staticmethod
    def _check_all_same(lst: List) -> bool:
        """
        检查列表中所有元素是否相同

        Args:
            lst: 元素列表

        Returns:
            bool: 所有元素是否相同
        """
        return all(x == lst[0] for x in lst)

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        从磁盘读取图像数据
        Args:
            image_fnames: 图像文件名列表
        Returns:
            Tuple[np.ndarray, dict]: 图像数组和元数据字典
        """
        pass

    @abstractmethod
    def read_segs(self, seg_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        从磁盘读取分割标签数据
        Args:
            seg_fnames: 分割标签文件名列表
        Returns:
            Tuple[np.ndarray, dict]: 分割标签数组和元数据字典
        """
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, seg_fname: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        将分割结果写入磁盘
        Args:
            seg: 分割结果数组
            seg_fname: 输出文件名
            metadata: 元数据字典（包含空间信息等）
        """
        pass



class SimpleITKReaderWriter(BaseReaderWriter):
    """SimpleITK格式读写器

    支持标准医学图像格式：NIfTI, MHA, MHD, NRRD等
    保留完整的医学图像元数据（间距、原点、方向等）
    """
    supported_file_formats = ["nii", "nii.gz", "mha", "mhd", "nrrd", "gz"]

    def __init__(self):
        super().__init__()

    def read_images(
        self, image_fnames: Union[str, list[str]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取SimpleITK支持的医学图像格式

        Args:
            image_fnames: 图像文件名列表

        Returns:
            Tuple[np.ndarray, Dict]: 图像数据和完整的元数据
        """

        if type(image_fnames) is not list:
            image_fnames = [image_fnames]

        image_data = []
        image_metadata = {"spacing": [], "origin": [], "direction": []}

        for image_fname in image_fnames:
            # 验证文件格式支持性
            if (
                os.path.basename(image_fname).split(".")[-1]
                not in self.supported_file_formats
            ):
                raise RuntimeError(f"File format not supported for {image_fname}")

            # 使用SimpleITK读取图像
            image = sitk.ReadImage(image_fname)
            logger.debug(f"Image {image_fname} has shape {image.GetSize()}")
            image_data.append(sitk.GetArrayFromImage(image))

            if image_data[-1].ndim != 3:
                raise RuntimeError(
                    f"Image {image_fname} has dimension {image_data[-1].ndim}, expected 3"
                )

            # 收集元数据信息
            image_metadata["spacing"].append(image.GetSpacing())
            image_metadata["origin"].append(image.GetOrigin())
            image_metadata["direction"].append(image.GetDirection())

        # 验证多个图像的元数据一致性
        if not self._check_all_same(image_metadata["spacing"]):
            logger.error("Spacing is not the same for all images")
            raise RuntimeError("Spacing is not the same for all images")
        if not self._check_all_same(image_metadata["origin"]):
            logger.warning("Origin is not the same for all images")
            logger.warning("Please check if this is expected behavior")
        if not self._check_all_same(image_metadata["direction"]):
            logger.warning("Direction is not the same for all images")
            logger.warning("Please check if this is expected behavior")

        # 提取统一的元数据
        sitk_metadata = {}
        for key in image_metadata.keys():
            sitk_metadata[key] = image_metadata[key][0]

        # 调整间距顺序以匹配numpy数组维度顺序 (z, y, x) -> (z, x, y)
        spacing = [
            sitk_metadata["spacing"][2],  # z-spacing
            sitk_metadata["spacing"][0],  # x-spacing
            sitk_metadata["spacing"][1],  # y-spacing
        ]
        meta_data = {"spacing": spacing, "other": sitk_metadata}
        logger.debug(f"Spacing: {spacing}")
        logger.debug(f"Final shape: {np.vstack(image_data).shape}")
        return np.vstack(image_data), meta_data

    def read_segs(
        self, seg_fnames: Union[str, list[str]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取分割标签数据"""
        return self.read_images(seg_fnames)

    def write_seg(
        self,
        seg: np.ndarray,
        seg_fname: str,
        metadata: Optional[Dict[str, Any]] = None,
        compression: bool = True,
    ):
        """写入分割结果到医学图像格式"""
        seg = sitk.GetImageFromArray(seg)
        if metadata is not None:
            # 恢复元数据信息
            seg.SetSpacing(metadata["spacing"])
            seg.SetOrigin(metadata["origin"])
            seg.SetDirection(metadata["direction"])

        sitk.WriteImage(seg, seg_fname, compression)


def determine_reader_writer(file_ending: str):
    """
    根据文件扩展名自动确定合适的读写器------------------------------------------------------------------------
    Args:
        file_ending: 文件扩展名
    Returns:
        BaseReaderWriter: 对应的读写器类
    """
    LIST_OF_READERS_WRITERS = [
        SimpleITKReaderWriter
    ]

    for reader_writer in LIST_OF_READERS_WRITERS:
        if file_ending.lower() in reader_writer.supported_file_formats:
            logger.debug(
                f"Automatically determined reader_writer: {reader_writer.__name__} for file ending: {file_ending}"
            )
            return reader_writer

    raise ValueError(f"No reader_writer found for file ending: {file_ending}")