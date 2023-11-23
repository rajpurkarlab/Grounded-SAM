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
import h5py
import pdb
import time

from utils import explore_tensor

class MIMICCXRDataset(Dataset):
    """MIMIC-CXR dataset."""

    def __init__(
        self, 
        csv_file='/n/data1/hms/dbmi/rajpurkar/lab/CXR-ReDonE/data/mimic_train_impressions.csv', 
        img_dir='/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files', 
        label_file='/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/2.0.0/mimic-cxr-2.0.0-chexpert.csv',
        h5_file='/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/initial_experiments/data/mimic_small.h5',
        size=(224,224), 
        tensor=False
    ):
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

        self.label_df = pd.read_csv(label_file)
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                        'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                        'Pneumothorax', 'Support Devices']
        
        self.img_dir = img_dir
        self.size = size
        self.tensor = tensor

        self.h5_file = h5_file

    def __len__(self):
        return 40
        # return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get report
        row = self.dataframe.iloc[idx]
        dicom_id = row['dicom_id']
        study_id = row['study_id']
        subject_id = row['subject_id']
        report = row['report']
        
        # Get image path
        image_path = f'{self.img_dir}/p{str(subject_id)[0:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'

        # Get labels: 1 if positive, 0 if negative, -1 if uncertain, nan if not mentioned
        label_row = self.label_df.loc[(self.label_df['subject_id'] == subject_id) & (self.label_df['study_id'] == study_id)]
        labels = label_row[self.classes]
        labels = labels.to_dict('records')[0]

        # Get preprocessed image and report from h5 file
        with h5py.File(self.h5_file,'r') as h5f:
            # Get processed images
            img_dset_gd = h5f['cxr_gd']
            img_gd = img_dset_gd[idx]
            img_dset_biovil = h5f['cxr_biovil']
            img_biovil = img_dset_biovil[idx]
            # Get report
            report_dset = h5f['report']
            report = report_dset[idx].decode()
            # Get image path
            image_path_dset = h5f['image_path']
            image_path = image_path_dset[idx].decode()

        if self.tensor:
            img = Image.open(image_path)
            # Resize image to self.size
            img = img.resize(self.size)
            convert_tensor = transforms.ToTensor()
            sample = {'image': convert_tensor(img), 'image_path': image_path, 'report': report, "labels": labels}
        else:
            sample = {'image_path': image_path, 'report': report, "labels": labels, "image_gd": img_gd, "image_biovil": img_biovil}
        
        return sample

def load_data(batch_size=16, tensor=False, num_workers=0):
    """Get dataloader for training.
    """
    dataset = MIMICCXRDataset(tensor=tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

class UnitTest:
    def __init__(self):
        pass
    
    def view_dataset(self):
        dataset = MIMICCXRDataset(tensor=False)
        for i in range(len(dataset)):
            print(len(dataset[i]["image_path"]))
            explore_tensor(dataset[i]["image_biovil"])

        # print(dataset[0])
        pdb.set_trace()

    def load_data_test(self):
        dataloader = load_data(tensor=False, num_workers=4)

        print("Number of batches:", len(dataloader))
        for i, data in enumerate(dataloader):
            # images = data["image"]
            image_paths = data["image_path"]
            report = data["report"]
            labels = data["labels"]
        
        print("Passed all tests")

if __name__=='__main__':
    unit_test = UnitTest()
    unit_test.view_dataset()