"""Training scripts for Grounded SAM.

Iteratively train between adaptation and segmentation objectives:
    - Adaptation: train Grounded SAM to align with frozen biomed CLIP using medical dataset.
    - Medical: train Grounded SAM to align the image and report embeddings using medical dataset.
    - Segmentation: train Grounded SAM with its original objective using natural dataset.
"""
import os
import pdb
import cv2
import time
import torch
import pycocotools.mask as mask_util
import json
import pandas as pd
import numpy as np
import h5py
import wandb
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import Dice
import torchvision.ops as ops
from PIL import Image
from transformers import SamProcessor
from info_nce import InfoNCE

from evaluation import eval_results
from utils import PROMPTS, explore_tensor

from model import myGroundingDino, myBiomedCLIP, mySAM, myBioViL
from dataset_mimic import load_data as load_mimic
from dataset_pascal import load_data as load_pascal

# seed
seed_value = 42
import random
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(hyperparams):
    """Train the model."""
    
    print("BOTH LOSSES, BIOVIL\n\n")

    # Load hyperparameters
    lr = hyperparams['lr']
    batch_size_ada = hyperparams['batch_size_adaptation']
    batch_size_seg = hyperparams['batch_size_segmentation']
    h5_file_mimic = hyperparams['h5_file_mimic']
    h5_file_pascal = hyperparams['h5_file_pascal']
    lambda_detection = hyperparams['lambda_detection']
    lambda_img_txt = hyperparams['lambda_img_txt']
    lambda_adaptation = hyperparams['lambda_adaptation']
    lambda_classif = hyperparams['lambda_classif']
    iou_thres = hyperparams['iou_thres']
    num_epochs = hyperparams['num_epochs']
    num_workers = hyperparams['num_workers']
    save_every = hyperparams['save_every']
    log_image_every = hyperparams['log_image_every']
    use_sam = hyperparams['use_sam']
    device = hyperparams['device']
    save_folder = hyperparams['save_folder']
    log_to_wandb = hyperparams['log_to_wandb']

    # Intialize wandb
    if log_to_wandb:
        run = wandb.init(
                entity="img-txt-localize",
                project="img-txt-localize",
                config={
                    "use_sam": use_sam,
                    "epochs": num_epochs,
                    "learning_rate": lr,
                    "batch_size_adaptation": batch_size_ada,
                    "batch_size_segmentation": batch_size_seg,
                }
            )

    # Load data
    mimic_dataloader = load_mimic(batch_size=batch_size_ada, h5_file=h5_file_mimic, num_workers=num_workers)
    pascal_dataloader = load_pascal(batch_size=batch_size_seg, h5_file=h5_file_pascal, num_workers=num_workers)
    pascal_dataloader_iter = inf_data_gen(pascal_dataloader)

    # Load model
    my_groundingdino = myGroundingDino(
        d=128,
        config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
        ckpt_file="./initial_experiments/ckpts/groundingdino_swint_ogc.pth",
        # ckpt_file="./initial_experiments/ckpts/initial_experiments_groundingdino_backbone_19695.pth",
        # img_linear_ckpt="./initial_experiments/ckpts/initial_experiments_groundingdino_img_linear_19695.pth",
        # txt_linear_ckpt="./initial_experiments/ckpts/initial_experiments_groundingdino_txt_linear_19695.pth",
        device=device,
    )
    my_groundingdino.model.backbone.train()
    my_groundingdino.img_linear.train()
    my_groundingdino.txt_linear.train()

    # my_biomedclip = myBiomedCLIP(device=device)
    # my_biomedclip.model.eval()
    my_biovil = myBioViL(device=device)
    my_biovil.model.eval()
    
    my_sam = None
    if use_sam:
        my_sam = mySAM(
            ckpt_file="./initial_experiments/ckpts/sam_vit_l_0b3195.pth",
            device=device,
        )
        my_sam.model.train()
        my_sam.img_linear.train()
    print("\nLoaded models!\n")
    
    # Load optimizer
    groundingdino_params = list(my_groundingdino.model.backbone.parameters()) + list(my_groundingdino.img_linear.parameters()) + list(my_groundingdino.txt_linear.parameters())
    if use_sam:
        sam_params = list(my_sam.model.parameters()) + list(my_sam.img_linear.parameters())
        optimizer = torch.optim.AdamW(groundingdino_params + sam_params, lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(groundingdino_params, lr=lr, weight_decay=1e-4)

    # Load scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=19000, T_mult=1, eta_min=1e-5)
    
    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(mimic_dataloader, desc=f'Training @ epoch {epoch+1} of {num_epochs}')):
            optimizer.zero_grad()

            # Compute adaptation loss on medical data (target domain)
            groundingdino_img_loss, groundingdino_txt_loss, groundingdino_img_txt_loss, sam_img_loss = compute_adaptation_loss(
                data, my_groundingdino, my_biovil, my_sam, on_dataset="mimic"
            )

            # Compute adaptation loss on pascal data (source domain)
            groundingdino_img_loss_s, groundingdino_txt_loss_s, groundingdino_img_txt_loss_s, sam_img_loss_s = compute_adaptation_loss(
                next(pascal_dataloader_iter), my_groundingdino, my_biovil, my_sam, on_dataset="pascal"
            )
            
            # Compute detection loss
            if i % log_image_every == 0:
                loss_detection = compute_detection_loss(next(pascal_dataloader_iter), my_groundingdino, step=i, iou_thres=iou_thres, viz=True, log_to_wandb=log_to_wandb)
            else:
                loss_detection = compute_detection_loss(next(pascal_dataloader_iter), my_groundingdino, step=i, iou_thres=iou_thres)
            
            # # start_time = time.time()
            # loss_classif = classification_loss(data, my_groundingdino, batch_size_seg)
            # end_time = time.time()
            # # print(f'Time taken: {end_time - start_time} seconds')
            
            loss = lambda_adaptation * (groundingdino_img_loss + groundingdino_txt_loss ) \
                        + lambda_img_txt * (groundingdino_img_txt_loss ) \
                        + lambda_detection * loss_detection \
                        + lambda_adaptation * (groundingdino_img_loss_s + groundingdino_txt_loss_s ) \
                        + lambda_img_txt * (groundingdino_img_txt_loss_s ) \
                        #   + lambda_classif * loss_classif 
            
            if i % log_image_every == 0:
                save_viz_chexlocalize(my_groundingdino, step=i, log_to_wandb=log_to_wandb)
            
            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    "train/loss": loss,
                    "train/groundingdino_img_loss": groundingdino_img_loss,
                    "train/groundingdino_txt_loss": groundingdino_txt_loss,
                    "train/groundingdino_img_txt_loss": groundingdino_img_txt_loss,
                    # "train/classif_loss": loss_classif,
                    "train/groundingdino_img_loss_s": groundingdino_img_loss_s,
                    "train/groundingdino_txt_loss_s": groundingdino_txt_loss_s,
                    "train/groundingdino_img_txt_loss_s": groundingdino_img_txt_loss_s,
                    "train/loss_detection": loss_detection,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                })
            
            # Training step
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Save model
            if i % (save_every+1) == 0 and i != 0:
                my_groundingdino.save_model(
                    ckpt_folder=save_folder,
                    backbone_ckpt=f"initial_experiments_groundingdino_backbone_{i}.pth",
                    img_linear_ckpt=f"initial_experiments_groundingdino_img_linear_{i}.pth",
                    txt_linear_ckpt=f"initial_experiments_groundingdino_txt_linear_{i}.pth",
                )
                if use_sam:
                    my_sam.save_model(
                        ckpt_folder=save_folder,
                        backbone_file=f"initial_experiments_sam_{i}.pth",
                        img_linear_ckpt=f"initial_experiments_sam_img_linear_{i}.pth",
                    )

                # Evaluation
                iou_pascal = eval_results(
                    "pascal", 
                    "grounded-sam", 
                    save_folder + f"initial_experiments_groundingdino_backbone_{i}.pth"
                )

                iou_chex = eval_results("chexlocalize", "grounded-sam", save_folder + f"initial_experiments_groundingdino_backbone_{i}.pth")

                auc_chexpert = eval_results(
                    "chexpert", "grounded-sam",
                    ckpt_file = save_folder + f"initial_experiments_groundingdino_backbone_{i}.pth", 
                    ckpt_img_linear = save_folder + f"initial_experiments_groundingdino_img_linear_{i}.pth",
                    ckpt_txt_linear = save_folder + f"initial_experiments_groundingdino_txt_linear_{i}.pth",
                )

                if log_to_wandb:
                    wandb.log({
                        "val/iou_pascal": iou_pascal, 
                        "val/iou_chex": iou_chex,
                        "val/auc_chexpert": auc_chexpert,
                        })

    # Finish wandb session
    wandb.finish()


def inf_data_gen(dataloader):
    """Infinite data generator.
    
    Used to loop through dataloader infinitely. We can access it by next(output).
    """
    while True:
        for data in dataloader:
            yield data


def compute_adaptation_loss(data, my_groundingdino, my_biovil, my_sam, on_dataset="mimic"):
    """Compute adaptation loss between grounding dino and biomedclip + between sam and biovil.
    
    Loss function: cosine similarity between embeddings if SAM is activated, InfoNCE loss if SAM is not activated.
    """
    # Load data
    image_gd = data["image_gd"].to(my_groundingdino.device)
    image_biovil = data["image_biovil"].to(my_biovil.device)
    if on_dataset=="mimic":
        reports = data["report"]
    elif on_dataset=="pascal":
        reports = [get_prompt(item) for item in data["labels"]]

    # Get embeddings
    groundingdino_img_emb = my_groundingdino.get_img_emb(image_gd)
    groundingdino_img_emb = my_groundingdino.align_img_emb(groundingdino_img_emb)
    groundingdino_txt_emb = my_groundingdino.get_txt_emb(reports)
    groundingdino_txt_emb = my_groundingdino.align_txt_emb(groundingdino_txt_emb)

    with torch.no_grad(): # medical model is frozen
        medical_img_embedding = my_biovil.get_img_emb(image_biovil)
        medical_txt_embedding = my_biovil.get_txt_emb(reports)
    
    if my_sam:
        sam_img_embedding = my_sam.get_img_emb(image_paths)
        sam_img_embedding = my_sam.align_img_emb(sam_img_embedding)

    # Define loss function
    if my_sam:
        loss_fn = F.cosine_similarity
    else:
        loss_fn = InfoNCE()

    # Get loss
    groundingdino_img_loss = loss_fn(groundingdino_img_emb, medical_img_embedding).mean()
    groundingdino_txt_loss = loss_fn(groundingdino_txt_emb, medical_txt_embedding).mean()
    groundingdino_img_txt_loss = loss_fn(groundingdino_img_emb, groundingdino_txt_emb).mean()

    sam_img_loss = torch.tensor(0.0)
    if my_sam:
        sam_img_loss = -loss_fn(sam_img_embedding, medical_img_embedding).mean()

    # Flip sign of loss if using cosine similarity
    if my_sam:
        groundingdino_img_loss = -groundingdino_img_loss
        groundingdino_txt_loss = -groundingdino_txt_loss
        groundingdino_img_txt_loss = -groundingdino_img_txt_loss
        sam_img_loss = -sam_img_loss
    
    # Return
    return (groundingdino_img_loss, groundingdino_txt_loss, groundingdino_img_txt_loss, sam_img_loss)


def classification_loss(data, my_groundingdino, batch_size):
    """Compute classification loss."""
    # Load data
    image_paths = data["image_path"]
    l = data["labels"]
        
    # Predict bounding box
    loss = 0
    for b in range(batch_size):
        pos = []
        neg = []
        for key in l:
            li = l[key].tolist()[b]
            if np.isnan(li) or li==0.:
                neg.append(key)
            elif li==1.:
                pos.append(key)
                
        prompt = ""
        for key in pos+neg:
            prompt += key + " . "
        prompt = prompt.strip()
                
        BOX_TRESHOLD = 0.05
        TEXT_TRESHOLD = 0.05
       

        # Get predicted bbox
        pred_bboxs, logits = my_groundingdino.inference(
            image_path=[image_paths[b]],
            caption=[prompt],
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )       
               
        b_loss = 0
               
        for i, c in enumerate(pos):
            b_loss += torch.nn.BCEWithLogitsLoss()(logits[0][i:i+1], torch.tensor([1.]).to(my_groundingdino.device))
        
        # Negative
        for i, c in enumerate(neg):
            b_loss += torch.nn.BCEWithLogitsLoss()(logits[0][i+len(pos):i+len(pos)+1], torch.tensor([0.]).to(my_groundingdino.device))
        
        loss += b_loss / len(pos+neg)
    
    return loss


def get_prompt(labels):
    """Get prompt for Grounding Dino.
    
    Args:
        - labels: dict of {phrase: label} pairs.
    
    Returns:
        - prompt: string of prompt.
    """
    keys = list(labels.keys())
    prompt = ""
    for key in keys:
        prompt += key + " . "
    return prompt.strip()


def compute_detection_loss(data, my_groundingdino, step, iou_thres=0.8, viz=False, log_to_wandb=False):
    """Compute detection loss."""
    # Define loss function
    obj_loss_fn = nn.BCELoss(reduction="mean")
    l1_loss_fn = nn.L1Loss(reduction="none")

    # Load data
    image_paths = data["image_path"]
    images = data["image_gd"].to(my_groundingdino.device)
    original_img_size = data["original_img_size"]    
    prompts = [get_prompt(item) for item in data["labels"]]
    labels = data["labels"]

    # Compute loss
    total_loss = 0
    B = len(images)

    # pred_bboxs is a list (length B) tensors, each tensor is shaped (num_prompts, 4)
    pred_bboxs, logits = my_groundingdino.inference(
        images, 
        prompts, 
        original_img_size,
        box_threshold=0.05,
        text_threshold=0.05,
        train_mode=True,
    )

    for b in range(B):
        # Filter for valid gt bboxs
        for i, p in enumerate(labels[b]):
            gt_bboxs_target = torch.tensor(labels[b][p], device='cuda')

            # Compute IoU between each pred_bbox and each gt_bbox
            ious = ops.box_iou(pred_bboxs[b][i], gt_bboxs_target)
            ious, gt_indices = ious.max(dim=1) # max iou overlap for each predicted bbox

            # Compute binary classification loss
            gt_label = ious > iou_thres
            if gt_label.sum() == 0: # Pick top prediction if no box > threshold
                max_iou, max_idx = ious.max(dim=0)
                gt_label[max_idx] = 1
            obj_loss = obj_loss_fn(logits[b][i], gt_label.float())

            # Comput regression loss
            gt_bboxs_target = gt_bboxs_target[gt_indices]
            giou_loss = ops.generalized_box_iou_loss(gt_bboxs_target, pred_bboxs[b][i], reduction="none")
            l1_loss = l1_loss_fn(gt_bboxs_target, pred_bboxs[b][i]).mean(dim=1) / 256 # normalize by image size
            reg_loss = (gt_label.float() * (2.0 * giou_loss + 5.0 * l1_loss)).sum() / gt_label.sum()

            # Compute total loss
            total_loss += 2.0 * obj_loss + reg_loss
        
    if viz:
        # Get prediction with highest logits for the last sample
        b = B - 1
        logit = logits[b][-1]
        max_logit, max_idx = logit.max(dim=0)
        logit_str = str(max_logit.tolist())[:4]

        bbox = pred_bboxs[b][-1][max_idx]
        bbox2 = gt_bboxs_target[max_idx]

        # Draw on image
        img = cv2.imread(image_paths[b])
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0,0,255), thickness=4) # red for prediction
        cv2.rectangle(img, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), color=(0,255,0), thickness=4) # green for gt
        
        # Save and log
        cv2.imwrite(f"./initial_experiments/images/{prompts[b]}_{logit_str}_step_{step}.jpg", img)
        if log_to_wandb:
            images = wandb.Image(
                Image.open(f"./initial_experiments/images/{prompts[b]}_{logit_str}_step_{step}.jpg"), 
                caption=f"{prompts[b]}_{logit_str}_step_{step}"
            )
            wandb.log({"pascal_images": images})

    return total_loss / B


def save_viz_chexlocalize(my_groundingdino, step, log_to_wandb=False):
    """Visualization for CheXlocalize."""
    # Randomly choose an image
    json_obj = json.load(open("datasets/chexlocalize/CheXlocalize/gt_segmentations_test.json"))
    h5_file = './initial_experiments/data/chexlocalize.h5'
    b = random.randint(0, len(json_obj)-1)
    obj = list(json_obj.keys())[b]
    filename = "datasets/chexlocalize/CheXpert/test/" + obj.replace("_", "/", (obj.count('_')-1)) + ".jpg"
    image_paths = [filename]
    with h5py.File(h5_file, "r") as h5f:
        image_gd = h5f['img_gd'][b]
        original_img_size = h5f['img_size'][b]
    image_gd = torch.from_numpy(image_gd).unsqueeze(0).to(my_groundingdino.device)
    
    # Choose a disease
    for query in json_obj[obj]:
        annots = json_obj[obj][query]
        if annots['counts'] != 'ifdl3': # ifdl3 denotes an empty ground-truth mask; skip over these test samples
            mask = mask_util.decode(annots)
            if mask.max() > 0: 
                break
    
    # Predict bounding box
    pred_bboxs, logits = my_groundingdino.inference(image_gd, [query], [original_img_size], box_threshold=0.05, text_threshold=0.05, train_mode=False)
    bbox = pred_bboxs[0][0]
    
    # Draw predicted bbox
    img = cv2.imread(image_paths[0])
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, start_point, end_point, color=(0,0,255), thickness=4)
    
    # Draw ground-truth mask
    color = (0,255,0)
    alpha = 0.3
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    image_combined = cv2.addWeighted(img, 1 - alpha, image_overlay, alpha, 0)
    
    # Save and log
    cv2.imwrite(f"./initial_experiments/images/{query}_step_{step}.jpg", image_combined)
    if log_to_wandb:
        images = wandb.Image(
            Image.open(f"./initial_experiments/images/{query}_step_{step}.jpg"), 
            caption=f"{query}_step_{step}"
        )
        wandb.log({"chex_images": images})


def compute_segmentation_loss(batch, sam_class, my_sam=None):
    """Compute segmentation loss if sam is activated, otherwise, compute loss based on bbox."""
    sam = sam_class.model
    
    for name, param in sam.named_parameters():
        if name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_loss = Dice().to(device)
    
    sam.train()
    loss = 0
    
    for i in range(len(batch["pixel_values"])):
        inputs = {}
        inputs["image"]=batch["pixel_values"][i].to(device)
        inputs["boxes"] = batch["input_boxes"][i].to(device)
        inputs["original_size"] = batch["ground_truth_mask"][i].shape[-2:]

        outputs = sam([inputs], multimask_output=False)[0]

        # compute loss
        predicted_masks = outputs["masks"].squeeze(1)
        ground_truth_masks = torch.tensor(batch["ground_truth_mask"][i])
        loss += seg_loss(predicted_masks.to("cuda"), ground_truth_masks.unsqueeze(0).to("cuda"))

    return loss


class UnitTest:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_adaptation_loss(self):
        sam = mySAM()
        groundingdino = myGroundingDino()
        biomedclip = myBiomedCLIP()
        
        loss = compute_adaptation_loss(
                    ["datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"],
                    ["Lung lesion"],
                    groundingdino,
                    biomedclip,
                    sam,
                )
        print(loss)
        print("Test adaptation loss passed!")

    def test_detection_loss(self):
        dataloader = load_pascal(batch_size=16)
        groundingdino = myGroundingDino()
        
        for i, data in enumerate(dataloader):
            print(compute_detection_loss(data, groundingdino))
            break
        print("Test detection loss passed!")
    
    def test_seg_loss(self):
        num_pascal_samples, dataloader = load_pascal(batch_size=2)
        sam = mySAM()
        
        for i, data in enumerate(dataloader):
            print(compute_segmentation_loss(data, sam))
            break
        print("Test segmentation loss passed!")

    def test_medical_loss(self):
        groundingdino = myGroundingDino()
        
        loss = compute_medical_loss(
                    ["datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"],
                    ["Lung lesion"],
                    groundingdino,
                )
        print(loss)
        print("Test medical loss passed!")
    
    def run_training(self):
        hyperparams = {
            "lr": 1e-4,
            "batch_size_adaptation": 16,
            "batch_size_segmentation": 16,
            "h5_file_mimic": "/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/initial_experiments/data/mimic.h5",
            "h5_file_pascal": "/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/initial_experiments/data/pascal_train.h5",
            "lambda_detection": 8,
            "lambda_img_txt": 2,
            "lambda_adaptation": 1,
            "lambda_classif": 1,
            "iou_thres": 0.3,
            "num_epochs": 3,
            "num_workers": 4,
            "use_sam": False,
            "save_every": 1000,
            "log_image_every": 100,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "save_folder": "./initial_experiments/ckpts/",
            "log_to_wandb": True,
        }
        train(hyperparams)
        

if __name__ == "__main__":
    unit_test = UnitTest()
    unit_test.run_training()
    # unit_test.test_adaptation_loss()
    # unit_test.test_detection_loss()
    # unit_test.test_seg_loss()
    # unit_test.test_medical_loss()