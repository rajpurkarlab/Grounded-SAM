"""Training scripts for Grounded SAM.

Iteratively train between adaptation and segmentation objectives:
    - Adaptation: train Grounded SAM to align with frozen biomed CLIP using medical dataset.
    - Segmentation: train Grounded SAM with its original objective using natural dataset.
"""

import torch
from torchvision import transforms
from torchmetrics.classification import Dice

from model import mySAM, myGroundingDino, myBiomedCLIP
from dataset_mimic import load_data as load_mimic
from dataset_pascal import load_data as load_pascal, get_len

def train(hyperparams):
    """Train the model."""

    # Load hyperparameters
    lr = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    num_epochs = hyperparams['num_epochs']
    num_workers = hyperparams['num_workers']
    device = hyperparams['device']
    save_folder = hyperparams['save_folder']

    # Load data
    len_mimic, mimic_dataloader = load_mimic(batch_size=batch_size, num_workers=num_workers)
    print(len_mimic / batch_size)
    print(get_len())
    print(int(get_len()/(len_mimic / batch_size)))
    pascal_dataloader = load_pascal(batch_size=int(get_len()/(len_mimic / batch_size)), num_workers=num_workers)

    # Load model
    my_groundingdino = myGroundingDino(
        config_file="./ckpts/GroundingDINO_SwinT_OGC.py",
        ckpt_file="./ckpts/groundingdino_swint_ogc.pth",
        device=device,
    )
    my_sam = mySAM(
        ckpt_file="./ckpts/sam_vit_l_0b3195.pth",
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
        for data in tqdm(mimic_dataloader, desc=f'Training @ epoch {epoch+1} of {num_epochs}'):
            # Load data
            image_paths = data["image_path"]
            report = data["report"]

            # Training step
            optimizer.zero_grad()
            loss = compute_segmentation_loss(next(it), my_sam) 
                                                                # + compute_adaptation_loss(image_paths, report, my_groundingdino, my_sam, my_biomedclip)
            loss.backward()
            optimizer.step()

    # Save model
    my_groundingdino.save_model(ckpt_folder=save_folder)
    my_sam.save_model(ckpt_folder=save_folder)
    
def compute_segmentation_loss(batch, sam_class):
    sam = sam_class.model
    
    for name, param in sam.named_parameters():
        if name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_loss = Dice().to(device)
    
    sam.train()
    loss = 0
    
    for i in range(len(batch)):
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

def compute_adaptation_loss(batch, pathologies, groundingdino, sam, biomedclip):
    """
    batch: list of image paths
    pathologies: list of pathologies
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bmi = []
    bmt = []
    si = []
    gdi = []
    gdt = []
    
    groundingdino_txt_emb = groundingdino.get_txt_emb(pathologies)
    
    for i in range(len(batch)):
        emb = groundingdino_txt_emb[i, :][None, :]
        gdt.append(groundingdino.align_txt_emb(emb).squeeze())
     
    for i, image_path in enumerate(batch):
        gd_img_emb = groundingdino.get_img_emb([image_path])
        sam_img_emb = sam.get_img_emb([image_path])
        
        with torch.no_grad():
            biomedclip_img_emb = biomedclip.get_img_emb([image_path])
            biomedclip_txt_emb = biomedclip.get_txt_emb(pathologies[i])
            bmi.append(biomedclip_img_emb.to(device).squeeze())
            bmt.append(biomedclip_txt_emb.to(device).squeeze())

        grounding_dino_emb_aligned = groundingdino.align_img_emb(gd_img_emb)
        gdi.append(grounding_dino_emb_aligned.squeeze())
        
        sam_emb_aligned = sam.align_img_emb(sam_img_emb)
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
        
    def test_adaptation_loss(self):
        sam = mySAM()
        groundingdino = myGroundingDino()
        biomedclip = myBiomedCLIP()
        
        loss = compute_adaptation_loss(
                    ["datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"],
                    ["Lung lesion"],
                    groundingdino,
                    sam,
                    biomedclip
                )
        
        print(loss)
    
    def test_seg_loss(self):
        from dataset_pascal import load_data
        dataloader = load_data()
        sam = mySAM()
        
        for i, data in enumerate(dataloader):
            print(compute_segmentation_loss(data, sam))
            break
    
    def run_training(self):
        hyperparams = {
            "lr": 1e-4,
            "batch_size": 16,
            "num_epochs": 1,
            "num_workers": 4,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "save_folder": "./ckpts/",
        }
        train(hyperparams)
        

if __name__ == "__main__":
    unit_test = UnitTest()
    unit_test.run_training()
    # unit_test.test_seg_loss()