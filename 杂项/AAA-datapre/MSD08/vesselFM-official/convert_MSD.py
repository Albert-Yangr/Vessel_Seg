import SimpleITK as sitk
import os
from monai.transforms import CropForegroundd, ScaleIntensityRangePercentilesd
import numpy as np
import json


def convert_sitk_image(image: sitk.Image):
    spacing = image.GetSpacing()
    spacing = [spacing[2], spacing[0], spacing[1]]
    return sitk.GetArrayFromImage(image), {"origin": image.GetOrigin(), "spacing": spacing, "direction": image.GetDirection()}


def calculate_metadata(array: np.ndarray):
    """
        This function will calculate the metadata of the image.
    """
    array = array.astype(np.float64)
    return {"max": array.max(), "min": array.min(), "mean": array.mean(), "std": array.std(), "shape": array.shape, "p95": np.percentile(array, 95), "p5": np.percentile(array, 5)}

def save_array(array: np.ndarray, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    for i in range(array.shape[0]):
        np.save(os.path.join(output_folder, f"{i:04d}.npy"), array[i])

def save_metadata(metadata: dict, output_folder: str):
    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def upsample(image, label, factor=4):
    # Upsample image in z direction by factor
    size = [
        label.GetSize()[0],
        label.GetSize()[1],
        int(label.GetSize()[2] * factor),
    ]
    label.SetSpacing([1, 1, factor])
    image.SetSpacing([1, 1, factor])

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing([1, 1, 1])
    resampler.SetOutputOrigin(label.GetOrigin())
    resampler.SetOutputDirection(label.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize(size)
    resampled_label = resampler.Execute(label)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([1, 1, 1])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize(size)

    resampled_image = resampler.Execute(image)

    return resampled_image, resampled_label


def convert_MSD(folder: str, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "imagesTs"), exist_ok=True)

    train_folder = "imagesTr"
    label_folder = "labelsTr"
    test_folder = "imagesTs"

    crop_foreground = CropForegroundd(keys=["image", "label"], source_key="label")
    scale_intensity = ScaleIntensityRangePercentilesd(
        keys=["image"], lower=20, upper=98, b_min=0, b_max=1, clip=True
    )

    for sample in os.listdir(os.path.join(folder, train_folder)):
        if sample.startswith("."):
            continue
        print(f"Converting {sample}...")
        image = sitk.ReadImage(os.path.join(folder, train_folder, sample))
        label = sitk.ReadImage(os.path.join(folder, label_folder, sample))
        image, label = upsample(image, label)
        array_img, metadata_img = convert_sitk_image(image)
        array_label, metadata_label = convert_sitk_image(label)
        array_label[array_label != 1] = 0
        array_img = array_img[None, ...]
        array_label = array_label[None, ...]
        res = crop_foreground({"image": array_img, "label": array_label})
        res = scale_intensity(res)
        array_img = res["image"][0]
        array_label = res["label"][0]
        metadata_img = metadata_img | calculate_metadata(array_img)
        save_array(
            array_img, os.path.join(out_folder, train_folder, sample.split(".")[0])
        )
        save_metadata(
            metadata_img, os.path.join(out_folder, train_folder, sample.split(".")[0])
        )

        save_array(
            array_label, os.path.join(out_folder, label_folder, sample.split(".")[0])
        )
        save_metadata(
            metadata_label, os.path.join(out_folder, label_folder, sample.split(".")[0])
        )

    for sample in os.listdir(os.path.join(folder, test_folder)):
        if sample.startswith("."):
            continue
        image = sitk.ReadImage(os.path.join(folder, test_folder, sample))
        array, metadata = convert_sitk_image(image)
        metadata = metadata | calculate_metadata(array)
        save_array(array, os.path.join(out_folder, test_folder, sample.split(".")[0]))
        save_metadata(
            metadata, os.path.join(out_folder, test_folder, sample.split(".")[0])
        )


import os
import argparse
import sys


def main(folder:str, out_folder:str):
    basename = os.path.basename(folder)
    print(f"basename of folder {folder} is {basename}.")
    if basename == "":
        basename = os.path.basename(os.path.dirname(folder))
        print(f"corrected basename of folder {folder} is {basename}.")

    if basename.lower() == "msd_task8":
        convert_MSD(folder, out_folder)



if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
    parser = argparse.ArgumentParser()
    # 修改为
    parser.add_argument("--input_folder", type=str,
                        default="/home/yangrui/Project/Base-model/datasets/MSD08/msd_task8")

    parser.add_argument("--output_folder", type=str,
                        default="/home/yangrui/Project/Base-model/datasets/MSD08/MSD-Official")
    args = parser.parse_args()
    print(args.input_folder, args.output_folder)
    main(args.input_folder, args.output_folder)