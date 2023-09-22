"""PASCAL dataset class and dataloader.
    
This file should also contains a quick test script to verify the dataloader works
to traverse through all the images in the dataset.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from linear_probe import LinearProbe
import torch
from torchvision import transforms
from PIL import Image

from utils import get_queries

class PASCALDataset(Dataset):
    """PASCAL VOC dataset."""

    def __init__(self, size=(224,224), tensor=True):
        self.train_id_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
        self.class_name_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/class_names.txt'
        self.img_folder_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/JPEGImages'
        self.gt_folder_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/SegmentationClass'
    
        # Load class names
        self.class_names = []
        for line in open(self.class_name_path, 'r'):
            self.class_names.append(line.strip())
        
        # Load val ids
        self.train_ids = []
        for line in open(self.train_id_path, 'r'):
            id = line.strip()
            self.train_ids.append(id)
        
        self.size = size
        self.tensor = tensor

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        id = self.train_ids[idx]
        img_path = self.img_folder_path + '/' + id + '.jpg'

        # load ground truth
        gt_path = self.gt_folder_path + '/' + id + '.png'
        
        gt_masks = get_queries(gt_path, self.size)

        if self.tensor:
            img = Image.open(img_path)
            img = img.resize(self.size)
            gt_img = Image.open(gt_path)
            gt_img = gt_img.resize(self.size)
            convert_tensor = transforms.ToTensor()
            sample = {'image': convert_tensor(img), 'image_path': img_path, 'gt_image': convert_tensor(gt_img), 'gt_image_path': gt_path, 'gt_masks': gt_masks}
        else:
            sample = {'image_path': img_path, 'gt_image_path': gt_path, 'gt_masks': gt_masks}
                
        return sample

def load_data(batch_size=16, tensor=False):
    """Get dataloader for training.
    """
    dataset = PASCALDataset(tensor=tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class UnitTest:
    def __init__(self):
        pass

    def load_data_test(self):
        dataloader = load_data(tensor=True)

        print("Number of batches:", len(dataloader))
        for i, data in enumerate(dataloader):
            images = data["image"]
            masks = data["gt_masks"]
        print("Passed all tests")

if __name__=='__main__':
    unit_test = UnitTest()
    unit_test.load_data_test()