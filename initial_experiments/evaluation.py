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
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import json
from uuid import uuid4
import pycocotools.mask as mask_util
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils import PROMPTS, get_iou, get_queries
from models.grounded_sam import run_grounded_sam, env_setup, load_models
from models.baselines import run_biovil
from segment_anything import build_sam_vit_h, build_sam_vit_l, SamPredictor

from model import myGroundingDino, myBiomedCLIP, mySAM, myCheXzero, myBioViL


def eval_results(
    dataset, model, ckpt_file="./initial_experiments/ckpts/groundingdino_swint_ogc.pth", 
    ckpt_img_linear=None, ckpt_txt_linear=None, linear=None,
    GRADCAM=False, use_sam=False
):
    if dataset == "chexlocalize":
        result = eval_chexlocalize(model, GRADCAM, ckpt_file, use_sam=use_sam) # mIoU
    elif dataset == "pascal":
        result = eval_pascal(model, GRADCAM, ckpt_file) # mIoU
    elif dataset == "chexpert":
        result = eval_chexpert(model, ckpt_file, ckpt_img_linear, ckpt_txt_linear) # mAUC
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    return result

def eval_pascal(model, GRADCAM, ckpt_file, use_sam=False):
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
        print(ckpt_file)
        groundingdino = myGroundingDino(
            d=512,
            config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
            ckpt_file=ckpt_file,
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
    for id in tqdm(val_ids):
        # load image
        img_path = img_folder_path + '/' + id + '.jpg'

        # load ground truth
        gt_path = gt_folder_path + '/' + id + '.png'
        gt_masks = get_queries(gt_path, Image.open(img_path).size)

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
                    if use_sam:
                        pred_mask = run_grounded_sam(img_path, text_prompt, groundingdino_model, sam_predictor)
                    else:
                        bbox, logits = groundingdino.predict([img_path], [text_prompt], box_threshold=0.0)
                        # bbox = bbox[0].detach().numpy().astype(int)
                        bbox = bbox[0].type(torch.int64)
                        pred_mask = np.zeros_like(gt_mask)
                        pred_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1 
                        
                        # import cv2
                        # # from torchvision.ops import box_convert
                        # # xyxy = box_convert(boxes=[bbox], in_fmt="cxcywh", out_fmt="xyxy").numpy()
                        # # print(xyxy)
                        
                        # print(text_prompt)
                        
                        # img = cv2.imread(img_path)
                        # start_point = (int(bbox[0]), int(bbox[1]))
                        # end_point = (int(bbox[2]), int(bbox[3]))
                        # cv2.rectangle(img, start_point, end_point, color=(0,255,0), thickness=4)
                        
                        # cv2.imwrite("example_with_bounding_boxes.jpg", img)
                        
                        
                        # plt.imsave('pred_mask.png', np.array(pred_mask), cmap=cm.gray)
                        # print("Done.")
                        # exit(0)
                    
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
                # print(iou_score)
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

def eval_chexlocalize(model, GRADCAM, ckpt_file, use_sam=False):
    # Load model
    print(ckpt_file)
    if model == "grounded-sam":
        env_setup()
        
        groundingdino = myGroundingDino(
            d=512,
            config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
            ckpt_file=ckpt_file,
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
                        # bbox = bbox[0].detach().numpy().astype(int)
                        bbox = bbox[0].type(torch.int64)
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


def eval_chexpert(model_name, ckpt_file, ckpt_img_linear, ckpt_txt_linear):
    # Load model
    print(ckpt_file)
    env_setup()
    if model_name == "grounded-sam":
        model = myGroundingDino(
            d=512,
            config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
            ckpt_file=ckpt_file,
            img_linear_ckpt=ckpt_img_linear,
            txt_linear_ckpt=ckpt_txt_linear,
        )
    elif model_name == "biomed-clip":
        model = myBiomedCLIP()
    elif model_name == "chexzero":
        model = myCheXzero()
    elif model_name == "biovil":
        model = myBioViL()
    else:
        raise NameError(f"Model {model_name} not supported")

    # Load CheXlocalize test set
    df = pd.read_csv("datasets/chexlocalize/CheXpert/test_labels.csv")
    classes = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    gt_results = {prompt: [] for prompt in classes}
    pred_results = {prompt: [] for prompt in classes}

    # Get text embedding
    txt_embeddings = {}
    neg_txt_embeddings = {}
    for query in gt_results.keys():
        # Positive prompt
        text_prompt = query.lower()
        # text_prompt = "Finding suggests " + query.lower()
        txt_embedding = model.get_txt_emb([text_prompt])
        if model_name == "grounded-sam":
            txt_embedding = model.align_txt_emb(txt_embedding)
        txt_embedding = txt_embedding / txt_embedding.norm(dim=-1, keepdim=True)
        txt_embeddings[query] = txt_embedding

        # Negative prompt
        neg_text_prompt = "no " + query.lower()
        # neg_text_prompt = "Finding suggests no " + query.lower()
        neg_txt_embedding = model.get_txt_emb([neg_text_prompt])
        if model_name == "grounded-sam":
            neg_txt_embedding = model.align_txt_emb(neg_txt_embedding)
        neg_txt_embedding = neg_txt_embedding / neg_txt_embedding.norm(dim=-1, keepdim=True)
        neg_txt_embeddings[query] = neg_txt_embedding
    
    # Loop through all test samples (pathology, image, ground-truth mask) tuples
    for i, row in tqdm(df.iterrows()):
        filename = "datasets/chexlocalize/CheXpert/" + row['Path']

        # Get image embedding 
        img_embedding = model.get_img_emb([filename])
        if model_name == "grounded-sam":
            img_embedding = model.align_img_emb(img_embedding)
        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
        
        # Loop through all pathologies in a test sample
        for query in row.keys():
            # Load class
            if query not in classes:
                continue
            
            # Load gt label
            gt_label = row[query]

            # Get predictions
            pos_logits = img_embedding @ txt_embeddings[query].T # (1, num_classes)
            pos_logits = np.squeeze(pos_logits.detach().cpu().numpy(), axis=0)
            
            neg_logits = img_embedding @ neg_txt_embeddings[query].T # (1, num_classes)
            neg_logits = np.squeeze(neg_logits.detach().cpu().numpy(), axis=0)
            
            sum_pred = np.exp(pos_logits) + np.exp(neg_logits)
            prob = np.exp(pos_logits) / sum_pred
            
            # pos_score = torch.matmul(img_embedding, txt_embeddings[query].T).detach().cpu()
            # neg_score = torch.matmul(img_embedding, neg_txt_embeddings[query].T).detach().cpu()
            # inner_product = torch.cat([pos_score, neg_score], dim=1)
            # prob = F.softmax(inner_product, dim=1)[:,0].item()

            # Append to results
            gt_results[query].append(gt_label)
            pred_results[query].append(prob)

    # Compute AUC for each pathology
    auc_results = {prompt: 0 for prompt in classes}
    for class_name in classes:
        AUC = roc_auc_score(gt_results[class_name], pred_results[class_name])
        auc_results[class_name] = AUC

    # Compute mean
    mAUC = np.mean(list(auc_results.values()))
    auc_results['mAUC'] = mAUC

    # Save to json
    json.dump(auc_results, open(f'chexpert_{model_name}.json', 'w'))
    
    return mAUC



class UnitTest:
    def __init__(self):
        pass

    def run_eval_scripts(self):
        # print("Starting Grounded-SAM, CheXlocalize...")
        # print("Grounded-SAM, CheXlocalize: ", eval_results("chexlocalize", "grounded-sam", use_sam=False))
        
        # print("Starting Grounded-SAM, PASCAL...")
        # print("Grounded-SAM, PASCAL - 202: ", eval_results("pascal", "grounded-sam"))
        
        # print(eval_results("chexlocalize", "grounded-sam"))
        # print("Grounded-SAM, CheXlocalize adaptation only - 303: ", eval_results("chexlocalize", "grounded-sam", "./initial_experiments/ckpts/initial_experiments_groundingdino_backbone_303.pth"))
        
        # print("Grounded-SAM, PASCAL adaptation only - 303: ", eval_results("pascal", "grounded-sam", "./initial_experiments/ckpts/initial_experiments_groundingdino_backbone_303.pth"))

        # print("Grounded-SAM, PASCAL - 6565: ", eval_results("pascal", "grounded-sam", "./initial_experiments/ckpts/initial_experiments_groundingdino_backbone_6565.pth"))
        
        # print("Starting BioViL, CheXlocalize, GRADCAM=False...")
        # print("BioViL, CheXlocalize, GRADCAM=False: ", eval_results("chexlocalize", "biovil"))
        
        # print("Starting BioViL, CheXlocalize, GRADCAM=True...")
        # print("BioViL, CheXlocalize, GRADCAM=True: ", eval_results("chexlocalize", "biovil", GRADCAM=True))
        
        # print("Starting BioViL, PASCAL, GRADCAM=False...")
        # print("BioViL, PASCAL, GRADCAM=False: ", eval_results("pascal", "biovil"))
        
        # print("Starting BioViL, PASCAL, GRADCAM=True...")
        # print("BioViL, PASCAL, GRADCAM=True: ", eval_results("pascal", "biovil", GRADCAM=True))

        print("Starting Grounded-SAM, CheXpert...")
        print("Grounded-SAM, CheXpert: ", eval_results(
            dataset = "chexpert", 
            model = "grounded-sam",
            ckpt_file = "./initial_experiments/ckpts/initial_experiments_groundingdino_backbone_19695.pth", 
            ckpt_img_linear = "./initial_experiments/ckpts/initial_experiments_groundingdino_img_linear_19695.pth",
            ckpt_txt_linear = "./initial_experiments/ckpts/initial_experiments_groundingdino_txt_linear_19695.pth",
        ))

        # print("Starting BiomedCLIP, CheXpert...")
        # print("BiomedCLIP, CheXpert: ", eval_results(
        #     dataset = "chexpert", 
        #     model = "biomed-clip",
        # ))

        # print("Starting CheXzero, CheXpert...")
        # print("CheXzero, CheXpert: ", eval_results(
        #     dataset = "chexpert", 
        #     model = "chexzero",
        # ))

        # print("Starting BioViL, CheXpert...")
        # print("BioViL, CheXpert: ", eval_results(
        #     dataset = "chexpert", 
        #     model = "biovil",
        # ))



def displace_results():
    import json
    import pandas as pd
    from tabulate import tabulate

    # Load results
    groundingdino_baseline = json.load(open('./chexpert_grounding_dino_baseline_1020.json'))
    groundingdino_190k = json.load(open('./chexpert_grounding_dino_19k_1020.json'))
    biomedclip_baseline = json.load(open('./chexpert_biomed_clip_baseline.json'))
    biovil_baseline = json.load(open('./chexpert_biovil_baseline.json'))
    chexzero_baseline = json.load(open('./chexpert_chexzero_baseline.json'))

    # Display results in a table
    df = pd.DataFrame({
        "BiomedCLIP (baseline)": biomedclip_baseline,
        "BioViL (baseline)": biovil_baseline,
        "CheXzero (baseline)": chexzero_baseline,
        "GroundingDINO (baseline)": groundingdino_baseline,
        "GroundingDINO (19k)": groundingdino_190k,
    })
    table = tabulate(df, headers='keys', tablefmt='grid')
    title = "AUC results on CheXpert test set"
    print(title.center(len(table.splitlines()[0])))
    print(table)


if __name__=='__main__':
    # unit_test = UnitTest()
    # unit_test.run_eval_scripts()
    displace_results()