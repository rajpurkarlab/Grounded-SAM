"""Training scripts for Grounded SAM.

Iteratively train between adaptation and segmentation objectives:
    - Adaptation: train Grounded SAM to align with frozen biomed CLIP using medical dataset.
    - Segmentation: train Grounded SAM with its original objective using natural dataset.
"""
import os
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
from PIL import Image
from transformers import SamProcessor

from model import myGroundingDino, myBiomedCLIP, mySAM
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

    # Load hyperparameters
    lr = hyperparams['lr']
    batch_size_ada = hyperparams['batch_size_adaptation']
    batch_size_seg = hyperparams['batch_size_segmentation']
    num_epochs = hyperparams['num_epochs']
    num_workers = hyperparams['num_workers']
    save_every = hyperparams['save_every']
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
    num_mimic_samples, mimic_dataloader = load_mimic(batch_size=batch_size_ada, tensor=False, num_workers=num_workers)
    num_pascal_samples, pascal_dataloader = load_pascal(batch_size=batch_size_seg, num_workers=num_workers)
    # print(num_mimic_samples / batch_size)
    # print(int(get_len()/(num_mimic_samples / batch_size)))
    # pascal_dataloader = load_pascal(batch_size=int(get_len()/(num_mimic_samples / batch_size)), num_workers=num_workers)

    # Load model
    my_groundingdino = myGroundingDino(
        config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
        ckpt_file="./initial_experiments/ckpts/groundingdino_swint_ogc.pth",
        device=device,
    )
    my_sam = mySAM(
        ckpt_file="./initial_experiments/ckpts/sam_vit_l_0b3195.pth",
        device=device,
    )
    my_biomedclip = myBiomedCLIP(device=device)

    # Load optimizer
    groundingdino_params = list(my_groundingdino.model.backbone.parameters()) + list(my_groundingdino.img_linear.parameters()) + list(my_groundingdino.txt_linear.parameters())
    sam_params = list(my_sam.model.parameters()) + list(my_sam.img_linear.parameters())
    optimizer = torch.optim.Adam(groundingdino_params + sam_params, lr=lr)

    # Set up training mode
    my_groundingdino.model.backbone.train()
    my_groundingdino.img_linear.train()
    my_groundingdino.txt_linear.train()
    my_sam.model.train()
    my_sam.img_linear.train()
    my_biomedclip.model.eval()
    print("\nLoaded models!\n")
    
    # Training loop
    for epoch in range(num_epochs):
        
        it = iter(pascal_dataloader)
        for i, data in enumerate(tqdm(mimic_dataloader, desc=f'Training @ epoch {epoch+1} of {num_epochs}')):
            optimizer.zero_grad()

            # Load data
            image_paths = data["image_path"]
            reports = data["report"]

            # Compte loss
            loss_adaptation, groundingdino_img_similarity, groundingdino_txt_similarity, sam_img_similarity = compute_adaptation_loss(
                image_paths, reports, my_groundingdino, my_biomedclip, my_sam
            )
            loss_segmentation = torch.tensor(0.0)
            loss_segmentation = compute_segmentation_loss(next(it), my_sam)
            loss = loss_adaptation + loss_segmentation

            # Log to wandb
            if log_to_wandb:
                wandb.log({
                        "loss": loss,
                        "loss_adaptation": loss_adaptation,
                        "loss_segmentation": loss_segmentation,
                        "groundingdino_img_similarity": groundingdino_img_similarity,
                        "groundingdino_txt_similarity": groundingdino_txt_similarity,
                        "sam_img_similarity": sam_img_similarity,
                    })
            
            # Training step
            loss.backward()
            optimizer.step()
            
            # Save model
            if i % (save_every+1) == 0:
                my_groundingdino.save_model(
                    ckpt_folder=save_folder,
                    backbone_ckpt=f"initial_experiments_groundingdino_backbone_{save_every}.pth",
                    img_linear_ckpt=f"initial_experiments_groundingdino_img_linear_{save_every}.pth",
                    txt_linear_ckpt=f"initial_experiments_groundingdino_txt_linear_{save_every}.pth",
                )
                if use_sam:
                    my_sam.save_model(
                        ckpt_folder=save_folder,
                        backbone_file=f"initial_experiments_sam_{save_every}.pth",
                        img_linear_ckpt=f"initial_experiments_sam_img_linear_{save_every}.pth",
                    )


        # Evaluation - TODO
        miou_train, miou_val = 0, 0
        if log_to_wandb:
            wandb.log({
                "miou_train": miou_train, 
                "miou_val": miou_val,
                })

    # Finish wandb session
    wandb.finish()


def compute_adaptation_loss(image_paths, reports, my_groundingdino, my_biomedclip, my_sam):
    """Compute adaptation loss between grounding dino and biomedclip + between sam and biomedclip."""
    # Get embeddings
    groundingdino_img_emb = my_groundingdino.get_img_emb(image_paths)
    groundingdino_img_emb = my_groundingdino.align_img_emb(groundingdino_img_emb)
    groundingdino_txt_emb = my_groundingdino.get_txt_emb(reports)
    groundingdino_txt_emb = my_groundingdino.align_txt_emb(groundingdino_txt_emb)

    with torch.no_grad(): # BiomedCLIP is kept frozen
        bmc_img_embedding = my_biomedclip.get_img_emb(image_paths)
        bmc_txt_embedding = my_biomedclip.get_txt_emb(reports)
    
    if my_sam:
        sam_img_embedding = my_sam.get_img_emb(image_paths)
        sam_img_embedding = my_sam.align_img_emb(sam_img_embedding)

    # Get loss - cosine similarity (consider using InfoNCE loss if we can have larger batch sizes)
    groundingdino_img_similarity = F.cosine_similarity(groundingdino_img_emb, bmc_img_embedding).mean()
    groundingdino_txt_similarity = F.cosine_similarity(groundingdino_txt_emb, bmc_txt_embedding).mean()
    sam_img_similarity = torch.tensor(0.0)
    if my_sam:
        sam_img_similarity = F.cosine_similarity(sam_img_embedding, bmc_img_embedding).mean()
    loss = -(groundingdino_img_similarity + groundingdino_txt_similarity + sam_img_similarity)
    return (loss, groundingdino_img_similarity, groundingdino_txt_similarity, sam_img_similarity)

    
def compute_segmentation_loss(batch, sam_class):
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
    
    def test_seg_loss(self):
        num_pascal_samples, dataloader = load_pascal(batch_size=2)
        sam = mySAM()
        
        for i, data in enumerate(dataloader):
            print(compute_segmentation_loss(data, sam))
            break
    
    def run_training(self):
        hyperparams = {
            "lr": 1e-4,
            "batch_size_adaptation": 1,
            "batch_size_segmentation": 2,
            "num_epochs": 1,
            "num_workers": 4,
            "use_sam": True,
            "save_every": 1000,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "save_folder": "./initial_experiments/ckpts/",
            "log_to_wandb": False
        }
        train(hyperparams)
        

if __name__ == "__main__":
    unit_test = UnitTest()
    unit_test.run_training()
    # unit_test.test_adaptation_loss()
    # unit_test.test_seg_loss()