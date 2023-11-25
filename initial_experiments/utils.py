"""Utility functions for the initial experiments.
"""
from PIL import Image
import numpy as np
from datasets import Dataset as Dataset
from datasets import Image as DImage

"""SAM eval utils"""

def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    return bbox
        
"""PASCAL eval utils"""

def get_label_from_num(num):
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse", "motorbike", "person", "plant", "sheep", "sofa", "train", "monitor"]
    return labels[num-1]

def get_queries(img_path, size):
    img = Image.open(img_path)
    img = img.resize(size)
    img = np.array(img)

    masks = {}
    uniq_vals = np.unique(img).tolist()
    
    try:
        uniq_vals.remove(0)
    except:
        pass
    
    try:
        uniq_vals.remove(255)
    except:
        pass

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
    "Support Devices": "Support devices",
    "No Finding": "No finding",
    "Fracture": "Findings suggesting a fracture",
}


def get_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def explore_tensor(tensor):
    """Explore a tensor's shape, mean, std, min, max."""
    print("-----------------------")
    print("Shape:", tensor.shape)
    print("Mean:", tensor.mean())
    print("Std:", tensor.std())
    print("Min:", tensor.min())
    print("Max:", tensor.max())
    print("----------------------")
