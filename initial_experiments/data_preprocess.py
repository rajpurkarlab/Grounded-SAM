import os
import sys
sys.path.extend(["../", "./"])

import pdb

import glob
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import h5py
import cv2
from typing import *
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from models.biovil import load_biovil_and_transform, remap_to_uint8


def img_to_hdf5(csv_path, img_dir, out_filepath, resolution_gd=256): 
    """
    Convert directory of images into a .h5 file given paths to all 
    images. 
    """
    # Load dataframe from csv file
    dataframe = pd.read_csv(csv_path)

    # Load transforms for preprocessing
    transform_gd = preprocess_gd(resolution_gd)
    transform_biovil = preprocess_biovil()

    # Convert images to h5
    print("Convert images to h5...")
    dset_size = len(dataframe)
    failed_images = []

    with h5py.File(out_filepath,'w') as h5f:
        # Initialize datasets
        img_dset_gd = h5f.create_dataset('cxr_gd', shape=(dset_size, 3, resolution_gd, resolution_gd))  
        img_dset_biovil = h5f.create_dataset('cxr_biovil', shape=(dset_size, 3, 448, 448))
        report_dset = h5f.create_dataset('report', dtype=h5py.string_dtype(), shape=(dset_size,))
        image_path_dset = h5f.create_dataset('image_path', dtype=h5py.string_dtype(), shape=(dset_size,))

        for idx in tqdm(range(len(dataframe))):
            # Get data from dataframe
            row = dataframe.iloc[idx]
            dicom_id = row['dicom_id']
            study_id = row['study_id']
            subject_id = row['subject_id']
            report = row['report']
            image_path = f'{img_dir}/p{str(subject_id)[0:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'

            # Save to dataset
            try: 
                # Save report
                report_dset[idx] = report

                # Save image_path
                image_path_dset[idx] = image_path

                # Load image
                image = Image.open(image_path).convert("RGB")
                image = np.asarray(image)

                # Transform grounding dino image
                image_gd = transform_gd(image)

                # Transform biovil image
                image = remap_to_uint8(image)
                image = Image.fromarray(image).convert("L")
                image_biovil = transform_biovil(image)

                # Save image to h5
                img_dset_gd[idx] = image_gd
                img_dset_biovil[idx] = image_biovil

            except Exception as e: 
                failed_images.append((image_path, e))

    print(failed_images)
    print(f"{len(failed_images)} / {dset_size} images failed to be added to h5.", failed_images)

    # Filter problematic entries
    print("Filtering problematic entries...")
    with h5py.File(out_filepath,'r') as h5f:
        # Get processed images
        img_dset_gd = h5f['cxr_gd']
        img_dset_biovil = h5f['cxr_biovil']
        # Get report
        report_dset = h5f['report']
        # Get image path
        image_path_dset = h5f['image_path']

        # Get indices of problematic entries
        indices = []
        for idx in tqdm(range(len(dataframe))):
            try:
                img_dset_gd[idx]
                img_dset_biovil[idx]
                report_dset[idx]
                image_path_dset[idx]
            except Exception as e:
                indices.append(idx)

        # Filter out problematic entries
        img_dset_gd = np.delete(img_dset_gd, indices, axis=0)
        img_dset_biovil = np.delete(img_dset_biovil, indices, axis=0)
        report_dset = np.delete(report_dset, indices, axis=0)
        image_path_dset = np.delete(image_path_dset, indices, axis=0)
    
    # Save filtered data to h5
    print("Saving filtered data to h5...")
    with h5py.File(out_filepath,'w') as h5f:
        # Initialize datasets
        img_dset_gd = h5f.create_dataset('cxr_gd', data=img_dset_gd)  
        img_dset_biovil = h5f.create_dataset('cxr_biovil', data=img_dset_biovil)
        report_dset = h5f.create_dataset('report', data=report_dset)
        image_path_dset = h5f.create_dataset('image_path', data=image_path_dset)


def preprocess_gd(resolution_gd=256):
    """Preprocess function for Grounding DINO."""
    transform = torchvision.transforms.Compose([
        # T.RandomResize([800], max_size=1333),
        # S.CenterCrop(800),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        torchvision.transforms.Resize((resolution_gd, resolution_gd)),
    ])
    return transform


def preprocess_biovil(config_file='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'):
    """Preprocess function for BioVil."""
    _, transform = load_biovil_and_transform()
    return transform


def filter_h5():
    print("Filtering problematic entries...")
    
    with h5py.File('./initial_experiments/data/mimic_small.h5', 'r') as h5f:
        # Identify indices of non-problematic entries
        valid_indices = []
        for idx in tqdm(range(len(h5f['cxr_gd']))):  # Assuming all datasets have the same length
            try:
                if h5f['cxr_gd'][idx].size > 0 and h5f['cxr_biovil'][idx].size > 0 \
                   and h5f['report'][idx].size > 0 and h5f['image_path'][idx].size > 0:
                    valid_indices.append(idx)
            except Exception as e:
                pass  # Skip adding this index if there's an error

    # Create a new HDF5 file with filtered data
    print("Saving filtered data to h5...")
    with h5py.File('./initial_experiments/data/mimic_small_filtered.h5', 'w') as h5f_new:
        with h5py.File('./initial_experiments/data/mimic_small.h5', 'r') as h5f_old:
            # Iterate over datasets
            for dset_name in ['cxr_gd', 'cxr_biovil', 'report', 'image_path']:
                # Read data for valid indices
                data = [h5f_old[dset_name][i] for i in valid_indices]
                
                # Create new dataset in new file
                h5f_new.create_dataset(dset_name, data=data)


if __name__ == "__main__":
    # cvs_path = '/n/data1/hms/dbmi/rajpurkar/lab/CXR-ReDonE/data/mimic_train_impressions.csv'
    # img_dir='/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files'

    # h5_out_path = './initial_experiments/data/mimic.h5'
    # img_to_hdf5(cvs_path, img_dir, h5_out_path)

    filter_h5()