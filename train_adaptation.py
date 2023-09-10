import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from linear_probe import LinearProbe
from vr_adapt import load_models

class MIMICCXRDataset(Dataset):
    """MIMIC-CXR dataset."""

    def __init__(self, csv_file='/n/data1/hms/dbmi/rajpurkar/lab/CXR-ReDonE/data/mimic_train_impressions.csv', img_dir='/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files', transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

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
                
        sample = {'image_path': image_path, 'report': report}
        
        return sample

def load_data(batch_size=16):
    """Get dataloader for training.
    """
    dataset = MIMICCXRDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_model():
    """Return image encoder, text encoder, and linear probes."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    groundingdino, sam, biomedclip, tokenizer, preprocess_train, preprocess_val = load_models()
    
    grounding_dino_input_dims = [
        [1, 256, 100, 100],
        [1, 512, 50, 50],
        [1, 1024, 25, 25],
    ]
    grounding_dino_linear = LinearProbe(
        grounding_dino_input_dims,
        512,
        device,
        )
    
    sam_input_dims = [
        [1, 256, 64, 64]
    ]
    sam_linear = LinearProbe(
        sam_input_dims, 
        512,
        device,
        )
    
    groundingdino_txt_dims = [
        [1, 195, 256]
    ]
    grounding_dino_linear_txt = LinearProbe(
        groundingdino_txt_dims,
        512,
        device,
        )

    return groundingdino, sam, biomedclip, tokenizer, preprocess_train, grounding_dino_linear, grounding_dino_linear_txt, sam_linear
    # NOTE: Linear probes still need to be tuned

def train(hyparams, output_path, model_paths):
    """Train the model."""
    dataloader = load_data()
    groundingdino, sam, biomedclip, tokenizer, preprocess_train, grounding_dino_linear, grounding_dino_linear_txt, sam_linear = load_model()
    # ...


if __name__ == "__main__":
    train()