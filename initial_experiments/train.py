"""Training scripts for Grounded SAM.

Iteratively train between adaptation and segmentation objectives:
    - Adaptation: train Grounded SAM to align with frozen biomed CLIP using medical dataset.
    - Medical: train Grounded SAM to align the image and report embeddings using medical dataset.
    - Segmentation: train Grounded SAM with its original objective using natural dataset.
"""
import os
import pdb
import cv2

import torch
import pandas as pd
import numpy as np
import wandb
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import Dice
import torchvision.ops as ops
from PIL import Image
from transformers import SamProcessor
from info_nce import InfoNCE

from evaluation import eval_results

from model import myGroundingDino, myBiomedCLIP, mySAM, myBioViL
from dataset_mimic import load_data as load_mimic
from dataset_martin_pascal import load_data as load_pascal

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
    lambda_detection = hyperparams['lambda_detection']
    lambda_img_txt = hyperparams['lambda_img_txt']
    lambda_adaptation = hyperparams['lambda_adaptation']
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
    mimic_dataloader = load_mimic(batch_size=batch_size_ada, tensor=False, num_workers=num_workers)
    pascal_dataloader = load_pascal(batch_size=batch_size_seg, num_workers=num_workers)
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
        optimizer = torch.optim.Adam(groundingdino_params + sam_params, lr=lr)
    else:
        optimizer = torch.optim.Adam(groundingdino_params, lr=lr)    

    # Load scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=19000, T_mult=1, eta_min=2e-5)
    
    # Training loop
    for epoch in range(num_epochs):
        
        for i, data in enumerate(tqdm(mimic_dataloader, desc=f'Training @ epoch {epoch+1} of {num_epochs}')):
            optimizer.zero_grad()

            # Load data
            image_paths = data["image_path"]
            reports = data["report"]

            # Compute loss
            groundingdino_img_loss, groundingdino_txt_loss, groundingdino_img_txt_loss, sam_img_loss = compute_adaptation_loss(
                image_paths, reports, my_groundingdino, my_biovil, my_sam,
            )
            
            if i % log_image_every == 0:
                loss_detection = compute_detection_loss(next(pascal_dataloader_iter), my_groundingdino, step=i, viz=True, log_to_wandb=log_to_wandb)
            else:
                loss_detection = compute_detection_loss(next(pascal_dataloader_iter), my_groundingdino, step=i)
            
            loss = lambda_adaptation * (groundingdino_img_loss + groundingdino_txt_loss) + lambda_img_txt * groundingdino_img_txt_loss + lambda_detection * loss_detection

            if i % log_image_every == 0:
                save_viz(data, my_groundingdino, step=i, log_to_wandb=log_to_wandb)
            
            # Log to wandb
            if log_to_wandb:
                wandb.log({
                    "train/loss": loss,
                    "train/groundingdino_img_loss": groundingdino_img_loss,
                    "train/groundingdino_txt_loss": groundingdino_txt_loss,
                    "train/groundingdino_img_txt_loss": groundingdino_img_txt_loss,
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
                iou_pascal = eval_results("pascal", "grounded-sam", f"./initial_experiments/ckpts_2/initial_experiments_groundingdino_backbone_{i}.pth")
                iou_chex = eval_results("chexlocalize", "grounded-sam", f"./initial_experiments/ckpts_2/initial_experiments_groundingdino_backbone_{i}.pth")
                auc_chexpert = eval_results(
                    "chexpert", "grounded-sam",
                    ckpt_file = f"./initial_experiments/ckpts_2/initial_experiments_groundingdino_backbone_{i}.pth", 
                    ckpt_img_linear = f"./initial_experiments/ckpts_2/initial_experiments_groundingdino_img_linear_{i}.pth",
                    ckpt_txt_linear = f"./initial_experiments/ckpts_2/initial_experiments_groundingdino_txt_linear_{i}.pth",
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


def compute_adaptation_loss(image_paths, reports, my_groundingdino, my_biovil, my_sam):
    """Compute adaptation loss between grounding dino and biomedclip + between sam and biomedclip.
    
    Loss function: cosine similarity between embeddings if SAM is activated, InfoNCE loss if SAM is not activated.
    """
    # Get embeddings
    groundingdino_img_emb = my_groundingdino.get_img_emb(image_paths)
    groundingdino_img_emb = my_groundingdino.align_img_emb(groundingdino_img_emb)
    groundingdino_txt_emb = my_groundingdino.get_txt_emb(reports)
    groundingdino_txt_emb = my_groundingdino.align_txt_emb(groundingdino_txt_emb)

    with torch.no_grad(): # BiomedCLIP is kept frozen
        medical_img_embedding = my_biovil.get_img_emb(image_paths)
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


def compute_detection_loss(data, my_groundingdino, step, viz=False, log_to_wandb=False):
    """Compute detection loss."""
    # Load data
    image_paths = data["image_path"]
    labels = data["label"]
    for i in range(len(labels)):
        labels[i] = f"An image of a {labels[i]}"
    gt_bboxs = data["bbox"].to(my_groundingdino.device)
    
    # Predict bounding box
    pred_bboxs, logits = my_groundingdino.predict(image_paths, labels, box_threshold=0.0)
    
    total_loss = 0
    for b in range(gt_bboxs.shape[0]):
        best_loss = np.inf
        best_box = -1
        bboxs = gt_bboxs[b]
        
        # Find gt bbox with the highest IoU (i.e., smallest loss) with prediction
        for idx, box in enumerate(bboxs):
            if (box != -1).any():
                loss = ops.generalized_box_iou_loss(box, pred_bboxs[b], reduction="mean")
                if loss < best_loss:
                    best_loss = loss
                    best_box = idx
        total_loss += best_loss
    
    if viz:
        # b = random.randint(0, gt_bboxs.shape[0]-1)
        b = gt_bboxs.shape[0]-1 # martin changed to ensure the "best_box" is meaningful
        bbox = pred_bboxs[b]     
        bbox2 = gt_bboxs[b][best_box]
        
        img = cv2.imread(image_paths[b])
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, start_point, end_point, color=(0,0,255), thickness=4) # red for prediction
        cv2.rectangle(img, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), color=(0,255,0), thickness=4) # green for gt
        
        cv2.imwrite(f"./initial_experiments/images/{labels[b]}_step_{step}.jpg", img)
        
        if log_to_wandb:
            images = wandb.Image(
                Image.open(f"./initial_experiments/images/{labels[b]}_step_{step}.jpg"), 
                caption=f"{labels[b]}_step_{step}"
            )
                    
            wandb.log({"pascal_images": images})
            
    return total_loss / gt_bboxs.shape[0]


def save_viz(data, my_groundingdino, step, log_to_wandb=False):
    """Visualization for MIMIC-CXR."""
    # Load data
    b = random.randint(0, len(data["image_path"])-1)
    image_paths = [data["image_path"][b]]
    labels = [data["report"][b]]
    
    # Predict bounding box
    pred_bboxs, logits = my_groundingdino.predict(image_paths, labels, box_threshold=0.0)
    
    bbox = pred_bboxs[0]
        
    img = cv2.imread(image_paths[0])
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, start_point, end_point, color=(0,0,255), thickness=4)
    
    cv2.imwrite(f"./initial_experiments/images/{labels[0]}_step_{step}.jpg", img)
    
    if log_to_wandb:
        images = wandb.Image(
            Image.open(f"./initial_experiments/images/{labels[0]}_step_{step}.jpg"), 
            caption=f"{labels[0]}_step_{step}"
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
            "lr": 2e-4,
            "batch_size_adaptation": 16,
            "batch_size_segmentation": 16,
            "lambda_detection": 8,
            "lambda_img_txt": 2,
            "lambda_adaptation": 1,
            "num_epochs": 3,
            "num_workers": 8,
            "use_sam": False,
            "save_every": 10,
            "log_image_every": 10,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "save_folder": "./initial_experiments/ckpts_2/",
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