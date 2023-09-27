"""Training scripts for Grounded SAM.

Iteratively train between adaptation and segmentation objectives:
    - Adaptation: train Grounded SAM to align with frozen biomed CLIP using medical dataset.
    - Segmentation: train Grounded SAM with its original objective using natural dataset.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
from transformers import SamProcessor

from linear_probe import LinearProbe
from model import load_model, preprocess_sam, preprocess_groundingdino_img
from groundingdino.util.misc import nested_tensor_from_tensor_list
from utils import get_bounding_box, SAMDataset


def train(hyparams, output_path, model_paths):
    """Train the model."""
    # Load data and model
    dataloader = load_data(tensor=True)
    groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear = load_model()
    
    optimizers = {}
    optimizers["groundingdino"] = torch.optim.Adam(groundingdino.parameters(), lr=hyparams["lr"])
    optimizers["sam"] = torch.optim.Adam(sam.parameters(), lr=hyparams["lr"])
    optimizers["groundingdino_img_linear"] = torch.optim.Adam(groundingdino_img_linear.parameters(), lr=hyparams["lr"])
    optimizers["groundingdino_txt_linear"] = torch.optim.Adam(groundingdino_txt_linear.parameters(), lr=hyparams["lr"])
    optimizers["sam_linear"] = torch.optim.Adam(sam.parameters(), lr=hyparams["lr"])
    
    # Training loop
    groundingdino.train()
    sam.train()
    biomedclip.eval()
    for epoch_num in range(hyparams["epochs"]):
        print("Epoch #{}".format(epoch_num))

        for i, data in enumerate(dataloader):
            print("Batch #{}".format(i))
            # Load data
            images = data["image"]
            image_paths = data["image_path"]
            reports = data["report"]

            for key, optimizer in optimizers.items():
                optimizer.zero_grad()

            # Compute loss
            loss = compute_adaptation_loss(image_paths, reports, groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear)
            loss.backward()
            
            for key, optimizer in optimizers.items():
                optimizer.step()
    
    return groundingdino, sam, groundingdino_img_linear, groundingdino_txt_linear

def compute_segmentation_loss(batch, sam):
    batch_images = batch["image"]
    gt_masks = batch["gt_mask"]
        
    print("got here")

    images = []
    labels = []
    for i, img in enumerate(batch_images):
        gt_mask=gt_masks[i]
        images.append(transforms.ToPILImage()(img))
        labels.append(gt_mask.numpy())
        # print(gt_mask.numpy().shape)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(images=images, labels=labels, processor=processor)
    
    for name, param in sam.named_parameters():
        # print(name)
        if name.startswith("prompt_encoder"):
            param.requires_grad_(False)
            # print("FALSE")
    
    from torch.optim import Adam
    import monai

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(sam.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    
    from tqdm import tqdm
    from statistics import mean
    # import torch
    from torch.nn.functional import threshold, normalize

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)

    sam.train()
    
    loss = 0
    
    for i in range(len(train_dataset)):
        example = train_dataset[i]
        # for k,v in example.items():
        #     print(k,v.shape)
        inputs = {}
        # inputs["multimodal_"]
        inputs["image"]=example["pixel_values"].to(device)
        inputs["boxes"] = example["input_boxes"].to(device)

        outputs = sam([inputs], multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = example["ground_truth_mask"].float().to("cuda")
        loss += seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

    return loss

def compute_adaptation_loss(batch, pathologies, groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear):
    """
    batch: list of image paths
    pathologies: list of pathologies
    """
    bmi = []
    bmt = []
    si = []
    gdi = []
    gdt = []

    tokenized = groundingdino.tokenizer(pathologies, padding="max_length", max_length=195, return_tensors="pt")
    (
        text_self_attention_masks,
        position_ids
    ) = generate_masks_with_special_tokens_and_transfer_map_nocate(
        tokenized, groundingdino.specical_tokens, groundingdino.tokenizer
    )

    if text_self_attention_masks.shape[1] > groundingdino.max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : groundingdino.max_text_len, : groundingdino.max_text_len
        ]
        position_ids = position_ids[:, : groundingdino.max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : groundingdino.max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : groundingdino.max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : groundingdino.max_text_len]

    if groundingdino.sub_sentence_present:
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids
    else:
        tokenized_for_encoder = tokenized

    bert_output = groundingdino.bert(**tokenized_for_encoder)  # bs, 195, 768
    groundingdino_txt_emb = groundingdino.feat_map(bert_output["last_hidden_state"]).to(device)  # bs, 195, d_model
    
    for i in range(len(batch)):
        emb = groundingdino_txt_emb[i, :][None, :]
        gdt.append(groundingdino_txt_linear(emb).squeeze())
     
    for i, image_path in enumerate(batch):
        samples = [preprocess_groundingdino_img(image_path)]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        groundingdino_img_emb, _ = groundingdino.backbone(samples)
        
        # TODO: need to bring sam outside of torch.no_grad but may run into GPU issue.
        sam_img = preprocess_sam(sam, image_path)
        print(sam_img.shape)
        sam_img_emb = sam.image_encoder(sam_img)[0][0]
        
        with torch.no_grad():
            img, txt = preprocess_biomedclip(preprocess_train, tokenizer, image_path, pathologies[i])
            biomedclip_img_emb, biomedclip_txt_emb, _ = biomedclip(img, txt)
            bmi.append(biomedclip_img_emb.to(device).squeeze())
            bmt.append(biomedclip_txt_emb.to(device).squeeze())
        
        gd_img_emb = []
        for emb in groundingdino_img_emb:
            gd_img_emb.append(emb.tensors.to(device))

        grounding_dino_emb_aligned = groundingdino_img_linear(gd_img_emb)
        gdi.append(grounding_dino_emb_aligned.squeeze())
        
        si_emb = [sam_img_emb[None, :].to(device)]
        sam_emb_aligned = sam_linear(si_emb)
        si.append(sam_emb_aligned.squeeze())
            
    bmi = torch.stack(bmi)
    bmt = torch.stack(bmt)
    si = torch.stack(si)
    gdi = torch.stack(gdi)
    gdt = torch.stack(gdt)
            
    path2list = {}
    path2list_t = {}
    bmif = []
    bmtf = []
    uniq = list(set(pathologies))
    for path in uniq:
        l = []
        t = []
        for i, p in enumerate(pathologies):
            if p == path:
                l.append(bmi[i])
                t.append(bmt[i])
        path2list[path] = l
        path2list_t[path] = t
        
    for i, path in enumerate(pathologies):
        l = []
        t = []
        for p in uniq:
            if p != path:
                l += path2list[p]
                t += path2list_t[p]
        bmif.append(torch.stack(l))
        bmtf.append(torch.stack(t))
    
    bmif = torch.stack(bmif)
    bmtf = torch.stack(bmtf)
    
    loss = InfoNCE(negative_mode='paired')
    loss_sam = loss(si, bmi, bmif)
    loss_groundingdino_img = loss(gdi, bmi, bmif)
    loss_groundingdino_txt = loss(gdt, bmt, bmtf)
    
    return loss_sam + loss_groundingdino_img + loss_groundingdino_txt

class UnitTest:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_adaptation_loss():
        groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear = load_model(self.device)
        
        loss = compute_adaptation_loss(
                    ["datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"],
                    ["Lung lesion"],
                    groundingdino,
                    sam,
                    biomedclip,
                    tokenizer,
                    preprocess_train,
                    grounding_dino_linear,
                    grounding_dino_linear_txt,
                    sam_linear
                )
        
        print(loss)
    
    def test_seg_loss():
        from dataset_pascal import load_data
        dataloader = load_data(tensor=True)
        groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear = load_model(predictor=False)
        
        for i, data in enumerate(dataloader):
            print(compute_segmentation_loss(data, sam))
            break
    
    def run_training():
        hyparams = {
            "lr": 1e-4,
            "epochs": 1,
        }
        train(hyparams, None, None)
        

if __name__ == "__main__":
    # unit_test = UnitTest()
    # unit_test.test_adaptation_loss()
    unit_test.test_seg_loss()
    