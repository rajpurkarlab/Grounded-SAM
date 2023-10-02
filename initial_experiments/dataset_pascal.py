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
from transformers import SamProcessor

from utils import get_queries, get_bounding_box

class PASCALDataset(Dataset):
    """PASCAL VOC dataset."""

    def __init__(self, size=(256,256)):
        """
        tensors (boolean): If true, return image as tensor. If false, don't return image (much faster).
        """
        self.train_id_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
        self.class_name_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/class_names.txt'
        self.img_folder_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/JPEGImages'
        self.gt_folder_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/SegmentationClass'
        self.processor = processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
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
        
        self.samples = []
        
        for idx in range(len(self.train_ids)):
            id = self.train_ids[idx]
            img_path = self.img_folder_path + '/' + id + '.jpg'

            gt_path = self.gt_folder_path + '/' + id + '.png'
            
            gt_masks = get_queries(gt_path, self.size)

            img = Image.open(img_path)
            img = img.resize(self.size)
                            
            for val in gt_masks:
                ground_truth_mask = gt_masks[val]
                
                prompt = get_bounding_box(ground_truth_mask)
            
                # prepare image and prompt for the model
                inputs = self.processor(img, input_boxes=[[prompt]], return_tensors="pt")

                # remove batch dimension which the processor adds by default
                inputs = {k:v.squeeze(0) for k,v in inputs.items()}

                # add ground truth segmentation
                inputs["ground_truth_mask"] = ground_truth_mask
                
                # inputs["image"] = img
                # inputs["image_path"] = img_path
                
                self.samples.append(inputs)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        return self.samples[idx]

def load_data(batch_size=16, num_workers=0):
    """Get dataloader for training.
    """
    dataset = PASCALDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_len():
    dataset = PASCALDataset()
    return len(dataset)

class UnitTest:
    def __init__(self):
        pass

    def load_data_test(self):
        dataloader = load_data()

        print("Number of batches:", len(dataloader))
        for i, data in enumerate(dataloader):
            images = data["image"]
            masks = data["gt_mask"]
        print("Passed all tests")

if __name__=='__main__':
    unit_test = UnitTest()
    unit_test.load_data_test()