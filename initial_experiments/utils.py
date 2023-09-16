"""Utility functions for the initial experiments.
"""
from PIL import Image
import numpy as np

"""PASCAL eval utils"""

def get_label_from_num(num):
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse", "motorbike", "person", "plant", "sheep", "sofa", "train", "monitor"]
    return labels[num-1]

def get_queries(img_path):
    img = np.array(Image.open(img_path))

    masks = {}
    uniq_vals = np.unique(img).tolist()

    try:
        uniq_vals.remove(0)
    except:
        print("no 0")
    
    try:
        uniq_vals.remove(255)
    except:
        print("no 255")

    for val in uniq_vals:
        masks[get_label_from_num(val)] = (img == val).astype(int)
            
    return masks

"""CheXlocalize eval utils"""

PROMPTS = { # Baseline prompts
    "Enlarged Cardiomediastinum": "Findings suggesting enlarged cardiomediastinum", 
    "Cardiomegaly": "Findings suggesting cardiomegaly", 
    "Edema": "Findings suggesting an edema",
    "Lung Lesion": "Findings suggesting lung lesion", 
    "Airspace Opacity": "Findings suggesting airspace opacity",
    "Consolidation": "Findings suggesting consolidation",
    "Atelectasis": "Findings suggesting atelectasis",
    "Pneumothorax": "Findings suggesting pneumothorax",
    "Pleural Effusion": "Findings suggesting pleural effusion",
    "Support Devices": "Support devices" 
}

def get_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score