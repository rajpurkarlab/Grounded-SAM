import os
import sys
sys.path.extend(["../", "./"])

import pdb

import glob
import numpy as np
import pandas as pd
import csv
import json
from tqdm import tqdm

from PIL import Image
import h5py
from typing import *

import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.datasets as datasets


from models.biovil import load_biovil_and_transform, remap_to_uint8


def get_image_paths_mimic(csv_path, img_dir, top_k=None):
    """Get the list of image paths for MIMIC."""
    # Load dataframe from csv file
    dataframe = pd.read_csv(csv_path)

    # Get image paths
    image_paths = []
    for idx in range(len(dataframe)):
        # Get data from dataframe
        row = dataframe.iloc[idx]
        dicom_id = row['dicom_id']
        study_id = row['study_id']
        subject_id = row['subject_id']
        image_path = f'{img_dir}/p{str(subject_id)[0:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        image_paths.append(image_path)

        if top_k is not None and idx == top_k:
            break

    return image_paths


def get_image_paths_pascal(root, year='2012', image_set='train'):
    """Get the list of image paths for PASCAL."""
    pascal = datasets.VOCDetection(
        root, 
        year=year, 
        image_set=image_set, 
        download=False, 
        transform=None, 
        target_transform=None
    )
    return pascal.images


def get_image_paths_chexlocalize(json_path, img_dir):
    """Get the list of image paths for CheXlocalize."""
    json_obj = json.load(open(json_path))

    image_paths = []
    for obj in json_obj:
        image_path = img_dir + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"
        image_paths.append(image_path)
    return image_paths


def get_image_paths_pascal_val(val_ids_path, img_dir):
    """Get the list of image paths for PASCAL val."""
    val_ids = open(val_ids_path, 'r').read().splitlines()
    image_paths = []
    for val_id in val_ids:
        image_path = img_dir + val_id + ".jpg"
        image_paths.append(image_path)
    return image_paths


def img_to_hdf5_general(image_path_list, out_filepath, resolution_gd=256, top_k=None): 
    """
    Convert directory of images into a .h5 file given paths to all 
    images. 
    """
    # Load transforms for preprocessing
    transform_gd = preprocess_gd(resolution_gd)
    transform_biovil = preprocess_biovil()

    # Convert images to h5
    print("Convert images to h5...")
    dset_size = len(image_path_list)
    failed_images = []

    with h5py.File(out_filepath,'w') as h5f:
        # Initialize datasets
        img_dset_gd = h5f.create_dataset('img_gd', shape=(dset_size, 3, resolution_gd, resolution_gd))  
        img_dset_biovil = h5f.create_dataset('img_biovil', shape=(dset_size, 3, 448, 448))
        img_size_dset = h5f.create_dataset('img_size', shape=(dset_size, 2))

        for idx, image_path in enumerate(tqdm(image_path_list)):
            # Save to dataset
            try: 
                # Load image
                image = Image.open(image_path).convert("RGB")
                image = np.asarray(image)

                # Save original image size
                img_size_dset[idx] = image.shape[:2]

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
            
            if top_k is not None and idx == top_k:
                break

    print(failed_images)
    print(f"{len(failed_images)} / {dset_size} images failed to be added to h5.", failed_images)


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


if __name__ == "__main__":
    # # MIMIC
    # cvs_path = '/n/data1/hms/dbmi/rajpurkar/lab/CXR-ReDonE/data/mimic_train_impressions.csv'
    # img_dir='/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files'
    # h5_path_mimic = './initial_experiments/data/mimic_small.h5'
    # mimic_image_paths = get_image_paths_mimic(cvs_path, img_dir, top_k=100)
    # img_to_hdf5_general(mimic_image_paths, h5_path_mimic, top_k=100)

    # # PASCAL
    # pascal_root = "/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/"
    # h5_path_pascal = './initial_experiments/data/pascal_train.h5'
    # pascal_image_paths = get_image_paths_pascal(root=pascal_root)
    # img_to_hdf5_general(pascal_image_paths, h5_path_pascal)

    # CheXlocalize
    json_path = "datasets/chexlocalize/CheXlocalize/gt_segmentations_test.json"
    img_dir = "datasets/chexlocalize/CheXpert/test/"
    h5_path_chexlocalize = './initial_experiments/data/chexlocalize.h5'
    chexlocalize_image_paths = get_image_paths_chexlocalize(json_path, img_dir)
    img_to_hdf5_general(chexlocalize_image_paths, h5_path_chexlocalize)
    
    # # PASCAL VAL
    # val_ids_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    # img_dir = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/JPEGImages/'
    # h5_path_pascal_val = './initial_experiments/data/pascal_val.h5'
    # pascal_image_paths = get_image_paths_pascal_val(val_ids_path, img_dir)
    # img_to_hdf5_general(pascal_image_paths, h5_path_pascal_val)

    # filter_h5()

