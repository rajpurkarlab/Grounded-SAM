import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from linear_probe import LinearProbe
import torch
from torchvision import transforms
from PIL import Image
import pickle
from transformers import SamProcessor

from utils import get_queries, get_bounding_box


class PASCALDataset(Dataset):
    """PASCAL VOC dataset."""

    def __init__(self):
        self.processed_data_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/Processed'
        

    def preprocess(self, size=(256,256)):
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
        
        for idx in tqdm(range(len(self.train_ids))):
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
                
                inputs["image"] = transforms.ToTensor()(img)
                inputs["image_path"] = img_path
                inputs["category"] = val

                # store to disk
                with open(os.path.join(self.processed_data_path, f'sample_{idx}.pkl'), 'wb') as file:
                    pickle.dump(inputs, file)

    
    def __len__(self):
        files = [f for f in os.listdir(self.processed_data_path)]
        return len(files)

    
    def __getitem__(self, idx):
        with open(os.path.join(self.processed_data_path, f'sample_{idx}.pkl'), 'rb') as file:
            data = pickle.load(file)
        return data


def load_data(batch_size=16, num_workers=0):
    """Get dataloader for training.
    """
    dataset = PASCALDataset()
    l = len(dataset)
    return l, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


class UnitTest:
    def __init__(self):
        pass

    def load_data_test(self):
        num_samples, dataloader = load_data(batch_size=4)

        print("Number of batches:", len(dataloader))
        for i, data in enumerate(tqdm(dataloader)):
            print(type(data))
        print("Passed all tests")
    
    def preprocess_test(self):
        dataset = PASCALDataset()
        dataset.preprocess()
        print("Passed preprocess")


if __name__=='__main__':
    unit_test = UnitTest()
    unit_test.preprocess_test()
    # unit_test.load_data_test()