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

my_groundingdino = myGroundingDino(
    config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
    ckpt_file="./initial_experiments/ckpts/groundingdino_swint_ogc.pth",
    device=device,
)

lr = 0.05

num_pascal_samples, pascal_dataloader = load_pascal(batch_size=batch_size_seg, num_workers=num_workers)

groundingdino_params = list(my_groundingdino.model.backbone.parameters())
optimizer = torch.optim.Adam(groundingdino_params, lr=lr)

# Set up training mode
my_groundingdino.model.backbone.train()
print("\nLoaded models!\n")

def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases

# Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(tqdm(pascal_dataloader, desc=f'Training @ epoch {epoch+1} of {num_epochs}')):
        optimizer.zero_grad()
        
        model = model.to(device)
        image = image.to(device)
        
        print(data)
        