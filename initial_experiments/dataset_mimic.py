"""MIMIC-CXR dataset class and dataloader
    
This file should also contains a quick test script to verify the dataloader works
to traverse through all the images and reports in the dataset.
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

class MIMICCXRDataset(Dataset):
    """MIMIC-CXR dataset."""

    def __init__(self, csv_file='/n/data1/hms/dbmi/rajpurkar/lab/CXR-ReDonE/data/mimic_train_impressions.csv', img_dir='/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files', size=(224,224), tensor=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            tensors (boolean): If true, return image as tensor. If false, don't return image (much faster).
        """
        self.dataframe = pd.read_csv(csv_file)
        self.dataframe.dropna(subset=['report'], inplace=True)
        
        self.img_dir = img_dir
        self.size = size
        self.tensor = tensor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        dicom_id = row['dicom_id']
        study_id = row['study_id']
        subject_id = row['subject_id']
        report = row['report']
        
        image_path = f'{self.img_dir}/p{str(subject_id)[0:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        
        if self.tensor:
            img = Image.open(image_path)
            # Resize image to self.size
            img = img.resize(self.size)
            convert_tensor = transforms.ToTensor()
            sample = {'image': convert_tensor(img), 'image_path': image_path, 'report': report}
        else:
            sample = {'image_path': image_path, 'report': report}
        
        return sample

def load_data(batch_size=16, tensor=False, num_workers=0):
    """Get dataloader for training.
    """
    dataset = MIMICCXRDataset(tensor=tensor)
    l = len(dataset)
    return l, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

class UnitTest:
    def __init__(self):
        pass

    def load_data_test(self):
        num_samples, dataloader = load_data(tensor=False)

        print("Number of batches:", len(dataloader))
        for i, data in enumerate(dataloader):
            # images = data["image"]
            image_paths = data["image_path"]
            report = data["report"]
        print("Passed all tests")

if __name__=='__main__':
    unit_test = UnitTest()
    unit_test.load_data_test()