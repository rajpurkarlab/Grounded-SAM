import os
import pandas as pd
from typing import Optional, Tuple
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = 'The value for percentiles should be a sequence of length 2,' f' but has length {len_percentiles}'
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)


class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def create_chest_xray_transform_for_inference(resize: int = 512, center_crop_size: int = 448) -> Compose:
    """
    Defines the image transformation pipeline for Chest-Xray datasets.

    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    """

    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)


class MSCXR(Dataset):
    def __init__(self, bucket_name, label_file, split, device, transform):
        # Set up GCS bucket
        self.bucket = storage.Client().bucket(bucket_name)
        self.prefix = os.path.join("ms_cxr", split)

        # Load label file
        blob = storage.Blob(label_file, self.bucket)
        data = blob.download_as_text()
        df = pd.read_csv(BytesIO(data.encode()))
        self.dataframe = df[df["split"] == split]
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get label
        row = self.dataframe.iloc[idx]
        text_prompt = row.label_text
        ground_truth_boxes = torch.tensor([
            row.x,
            row.y, 
            row.w,
            row.h,
        ])

        # Get image
        image_blob = storage.Blob(f"{self.prefix}/{row.dicom_id}.jpg", self.bucket)
        image_data = image_blob.download_as_bytes()
        image = Image.open(BytesIO(image_data))
        image = np.array(image)
        image = remap_to_uint8(image)
        image = Image.fromarray(image).convert("L")
        transformed_image = self.transform(image)

        # Combine
        data = {
            "image": transformed_image,
            "text": text_prompt,
            "ground_truth_boxes": ground_truth_boxes,
            "dicom_id": f"{row.dicom_id}.jpg",
        }
        return data

    def visualize(self, idx):
        """Visualize an image with boudning boxes after transformation."""
        data = self.__getitem__(idx)
        image = data["image"]
        text = data["text"]
        ground_truth_boxes = data["ground_truth_boxes"]
        image = image.squeeze(0)
        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.array(image)

        # Overlay bounding boxes
        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap="gray")
        rect = patches.Rectangle(
            (ground_truth_boxes[0], ground_truth_boxes[1]),
            ground_truth_boxes[2],
            ground_truth_boxes[3],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_title(text)
        plt.savefig("test.png")
        return fig
    

def get_mscxr_dataloader(split, batch_size, device):
    dataset = MSCXR(
        bucket_name="radiq-app-data",
        label_file="ms_cxr/label_1024_split.csv",
        split=split,
        device=device,
        transform=create_chest_xray_transform_for_inference(),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class UnitTest:
    """Unit test for MSCXR dataset."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dataset(self):
        resize = 512
        dataset = MSCXR(
            bucket_name="radiq-app-data",
            label_file="ms_cxr/label_1024_split.csv",
            split="train",
            device=self.device,
            resize=resize,
            transform=create_chest_xray_transform_for_inference(),
        )
        # for i, data in enumerate(tqdm(dataset)):
        #     # print(data)
        #     pass
        # print("Test MSCXR dataset: SUCCESS!")
        return dataset

    def test_dataloader(self):
        split = "train"
        batch_size = 16
        device = self.device
        dataloader = get_mscxr_dataloader(split, batch_size, device)
        for batch_idx, data in enumerate(tqdm(dataloader)):
            # print(data)
            pass 
        print("Test MSCXR dataloader: SUCCESS!")
    
    def test_visualize(self):
        dataset = self.test_dataset()
        dataset.visualize(10)
        print("Test MSCXR visualize: SUCCESS!")
    

if __name__ == "__main__":
    test = UnitTest()
    test.test_dataloader()