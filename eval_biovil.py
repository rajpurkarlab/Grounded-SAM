import json
from uuid import uuid4
import pycocotools.mask as mask_util
import argparse
import sys
import numpy as np

GRADCAM = True

from models.baselines import run_biovil

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

def chexlocalize_eval():
    # Evaluation
    iou_results = {prompt: [] for prompt in PROMPTS}

    json_obj = json.load(open("datasets/chexlocalize/CheXlocalize/gt_segmentations_test.json")) # Load CheXlocalize test set

    for obj in json_obj: # Loop through all test samples: (pathology, image, ground-truth mask) tuples
        filename = "datasets/chexlocalize/CheXpert/test/" + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"
        for query in json_obj[obj]:
            if query not in PROMPTS:
                continue
            annots = json_obj[obj][query]

            if annots['counts'] != 'ifdl3': # ifdl3 denotes an empty ground-truth mask; skip over these test samples
                gt_mask = mask_util.decode(annots) # Decode ground-truth mask using pycocotools library
                if gt_mask.max() == 0:
                    continue

                text_prompt = PROMPTS[query]

                if GRADCAM:
                    pred_mask = run_biovil(filename, text_prompt, gradcam=True)
                else:
                    pred_mask = run_biovil(filename, text_prompt)
                
                # compute iou
                try:
                    iou_score = get_iou(pred_mask, gt_mask)
                except:
                    iou_score = get_iou(pred_mask, np.swapaxes(gt_mask,0,1))

                print(filename, text_prompt, iou_score)

                iou_results[query].append(iou_score)
    
    # Compute and print pathology-specific mIoUs
    total_sum = 0
    total_count = 0
    for value_list in iou_results.values():
        total_sum += sum(value_list)
        total_count += len(value_list)
    mIoU = total_sum / total_count

    # compute mIoU by class and save to json
    mIoU_classes = {}
    for class_name in PROMPTS:
        mIoU_classes[class_name] = np.mean(iou_results[class_name])
    mIoU_classes['mIoU'] = mIoU
    # FNAME = ""
    json.dump(mIoU_classes, open('chexlocalize_biovil_gradcam_plus.json', 'w'))
    
    return mIoU

if __name__ == "__main__":
    mIoU = chexlocalize_eval()
    print(mIoU)