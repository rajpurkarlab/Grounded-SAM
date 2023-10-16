"""Evaluation script for Grounded SAM and baselines on different datasets.
    - Datasets: CheXlocalize, PASCAL TEST.
    - Models: Grounded SAM untrained, Grounded SAM trained, BioViL.

Main function is eval_results = func(dataset, model, [args])
If significant changes are need for different datasets, we should put them in helper functions,
and call them in the main function.
"""

import warnings
warnings.simplefilter("ignore")
import sys
sys.path.extend(["../", "./"])
import pdb

import json
from uuid import uuid4
import pycocotools.mask as mask_util
import argparse
import sys
import numpy as np
from tqdm import tqdm

from utils import PROMPTS, get_iou, get_queries
from models.grounded_sam import run_grounded_sam, env_setup, load_models
from models.baselines import run_biovil
from segment_anything import build_sam_vit_h, build_sam_vit_l, SamPredictor

from model import myGroundingDino, myBiomedCLIP, mySAM

def eval_results(dataset, model, GRADCAM=False, use_sam=False):
    if dataset == "chexlocalize":
        mIoU = eval_chexlocalize(model, GRADCAM, use_sam=use_sam)
    elif dataset == "pascal":
        mIoU = eval_pascal(model, GRADCAM)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    return mIoU

def eval_pascal(model, GRADCAM):
    # Specify paths
    val_id_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    class_name_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/class_names.txt'
    img_folder_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/JPEGImages'
    gt_folder_path = '/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/VOCdevkit/VOC2012/SegmentationClass'

    # Load class names
    class_names = []
    for line in open(class_name_path, 'r'):
        class_names.append(line.strip())
    
    # Load val ids
    val_ids = []
    for line in open(val_id_path, 'r'):
        id = line.strip()
        val_ids.append(id)

    # Load model
    if model == "grounded-sam":
        env_setup()
        groundingdino = myGroundingDino(
            config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
            ckpt_file="./initial_experiments/ckpts/groundingdino_swint_ogc.pth",
        )
        groundingdino_model = groundingdino.model

        sam = build_sam_vit_l(
            checkpoint="./initial_experiments/ckpts/sam_vit_l_0b3195.pth"
        ).to("cuda")
        sam_predictor = SamPredictor(sam)
    elif model == "biovil":
        pass
    else:
        raise NotImplementedError(f"Model {model} not supported")

    # Evaluation
    iou_results = {class_name: [] for class_name in class_names}
    for id in val_ids:
        # load image
        img_path = img_folder_path + '/' + id + '.jpg'

        # load ground truth
        gt_path = gt_folder_path + '/' + id + '.png'
        gt_masks = get_queries(gt_path)

        # run through all classes
        for class_name in class_names:
            if class_name not in gt_masks:
                continue

            # load gt mask
            gt_mask = gt_masks[class_name]

            # run grounded sam
            text_prompt = 'a ' + class_name

            try:
                if model == "grounded-sam":
                    pred_mask = run_grounded_sam(img_path, text_prompt, groundingdino_model, sam_predictor)
                elif model == "biovil":
                    if GRADCAM:
                        pred_mask = run_biovil(img_path, text_prompt, gradcam=True)
                    else:
                        pred_mask = run_biovil(img_path, text_prompt)
                else:
                    raise NotImplementedError(f"Model {model} not supported")   
                
                pred_mask = (pred_mask != 0).astype(int)

                # compute iou
                try:
                    iou_score = get_iou(pred_mask, gt_mask)
                except:
                    iou_score = get_iou(pred_mask, np.swapaxes(gt_mask,0,1))
                
                iou_results[class_name].append(iou_score)
            except:
                print(f"\nSkipping {img_path}, {text_prompt} due to errors\n")

    # compute average mIoU across all classes
    total_sum = 0
    total_count = 0
    for value_list in iou_results.values():
        total_sum += sum(value_list)
        total_count += len(value_list)
    mIoU = total_sum / total_count

    # compute mIoU by class and save to json
    mIoU_classes = {}
    for class_name in class_names:
        mIoU_classes[class_name] = np.mean(iou_results[class_name])
    mIoU_classes['mIoU'] = mIoU
    json.dump(mIoU_classes, open(f'pascal_{model}_gradcam={GRADCAM}.json', 'w'))
    
    return mIoU

def eval_chexlocalize(model, GRADCAM, use_sam=False):
    # Load model
    if model == "grounded-sam":
        env_setup()
        groundingdino = myGroundingDino(
            config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
            # ckpt_file="./initial_experiments/ckpts/groundingdino_swint_ogc.pth",
            ckpt_file="./initial_experiments/ckpts/initial_experiments_groundingdino_backbone_100.pth",
        )
        groundingdino_model = groundingdino.model

        if use_sam:
            sam = build_sam_vit_l(
                checkpoint="./initial_experiments/ckpts/sam_vit_l_0b3195.pth",
                # checkpoint="./initial_experiments/ckpts/initial_experiments_sam_1000.pth",
            ).to("cuda")
            sam_predictor = SamPredictor(sam)
    elif model == "biovil":
        pass
    else:
        raise NotImplementedError(f"Model {model} not supported")

    # Load CheXlocalize test set
    json_obj = json.load(open("datasets/chexlocalize/CheXlocalize/gt_segmentations_test.json"))

    iou_results = {prompt: [] for prompt in PROMPTS}
    # Loop through all test samples (pathology, image, ground-truth mask) tuples
    for obj in tqdm(json_obj):
        filename = "datasets/chexlocalize/CheXpert/test/" + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"
        
        # Loop through all pathologies in a test sample
        for query in json_obj[obj]:
            if query not in PROMPTS:
                continue
            annots = json_obj[obj][query]

            if annots['counts'] != 'ifdl3': # ifdl3 denotes an empty ground-truth mask; skip over these test samples
                gt_mask = mask_util.decode(annots) # Decode ground-truth mask using pycocotools library
                if gt_mask.max() == 0:
                    continue

                # Get predicted mask
                text_prompt = PROMPTS[query]
                if model == "grounded-sam":
                    if use_sam:
                        pred_mask = run_grounded_sam(filename, text_prompt, groundingdino_model, sam_predictor)
                    else:
                        bbox, logits = groundingdino.predict([filename], [text_prompt], box_threshold=0.0)
                        bbox = bbox[0].detach().numpy().astype(int)
                        pred_mask = np.zeros_like(gt_mask)
                        pred_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
                elif model == "biovil":
                    if GRADCAM:
                        pred_mask = run_biovil(filename, text_prompt, gradcam=True)
                    else:
                        pred_mask = run_biovil(filename, text_prompt)
                else:
                    raise NotImplementedError(f"Model {model} not supported")        
        
                pred_mask = (pred_mask != 0).astype(int)
                
                # Compute iou
                try:
                    iou_score = get_iou(pred_mask, gt_mask)
                except:
                    iou_score = get_iou(pred_mask, np.swapaxes(gt_mask,0,1))

                iou_results[query].append(iou_score)
    
    # Compute and print pathology-specific mIoUs
    total_sum = 0
    total_count = 0
    for value_list in iou_results.values():
        total_sum += sum(value_list)
        total_count += len(value_list)
    mIoU = total_sum / total_count

    # Compute mIoU by class and save to json
    mIoU_classes = {}
    for class_name in PROMPTS:
        mIoU_classes[class_name] = np.mean(iou_results[class_name])
    mIoU_classes['mIoU'] = mIoU
    json.dump(mIoU_classes, open(f'chexlocalize_{model}_sam={use_sam}.json', 'w'))
    
    return mIoU


class UnitTest:
    def __init__(self):
        pass

    def run_eval_scripts(self):
        print("Starting Grounded-SAM, CheXlocalize...")
        print("Grounded-SAM, CheXlocalize: ", eval_results("chexlocalize", "grounded-sam", use_sam=False))
        
        # print("Starting Grounded-SAM, PASCAL...")
        # print("Grounded-SAM, PASCAL: ", eval_results("pascal", "grounded-sam"))
        
        # print("Starting BioViL, CheXlocalize, GRADCAM=False...")
        # print("BioViL, CheXlocalize, GRADCAM=False: ", eval_results("chexlocalize", "biovil"))
        
        # print("Starting BioViL, CheXlocalize, GRADCAM=True...")
        # print("BioViL, CheXlocalize, GRADCAM=True: ", eval_results("chexlocalize", "biovil", GRADCAM=True))
        
        # print("Starting BioViL, PASCAL, GRADCAM=False...")
        # print("BioViL, PASCAL, GRADCAM=False: ", eval_results("pascal", "biovil"))
        
        # print("Starting BioViL, PASCAL, GRADCAM=True...")
        # print("BioViL, PASCAL, GRADCAM=True: ", eval_results("pascal", "biovil", GRADCAM=True))


if __name__=='__main__':
    unit_test = UnitTest()
    unit_test.run_eval_scripts()