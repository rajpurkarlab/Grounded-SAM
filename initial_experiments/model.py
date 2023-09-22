"""Model and preprocess function loading for grounded SAM experiment.
"""
import warnings
warnings.simplefilter("ignore")
import sys
sys.path.extend(["../", "./"])

import torch
import torchvision
from PIL import Image
import cv2
import open_clip
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry, build_sam, SamPredictor

from models.grounded_sam import *
from models.GroundingDINO.groundingdino.util.inference import load_image
from linear_probe import LinearProbe


def load_model(device="cuda", predictor=False):
    """Load image encoders, text encoders, and linear probes.
    
    Load Grounding Dino - model(image encoder, text encoder), linear probe for image embedding and text embedding.
         SAM - model(image encoder), linear probe for image embedding
         Biomed CLIP - model(image encoder, text encoder), tokenizer for text, preprocess for image
    """
    # Load Grounding Dino
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename) # groundingdino.backbone, groundingdino.bert, groundingdino.tokenizer
    
    # Load Grounded SAM
    sam_checkpoint = './initial_experiments/ckpts/sam_vit_l_0b3195.pth'
    sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint)
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    # Load Biomed CLIP
    biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    biomedclip.to(device)

    # Load linear probe for Grounding Dino image embedding
    groundingdino_input_dims = [
        [1, 256, 100, 100],
        [1, 512, 50, 50],
        [1, 1024, 25, 25],
    ]
    groundingdino_img_linear = LinearProbe(
        groundingdino_input_dims,
        512,
        device,
    )

    # Load linear probe for Grounding Dino text embedding
    groundingdino_txt_dims = [
        [1, 195, 256]
    ]
    groundingdino_txt_linear = LinearProbe(
        groundingdino_txt_dims,
        512,
        device,
    )
    
    # Load linear probe for SAM image embedding
    sam_input_dims = [
        [1, 256, 64, 64]
    ]
    sam_linear = LinearProbe(
        sam_input_dims, 
        512,
        device,
    )
    if predictor:
        return groundingdino, sam_predictor, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear

    return groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear


def preprocess_sam(sam, image_path, device="cuda"):
    """Preprocess image for SAM."""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((20, 20)),
        torchvision.transforms.ToTensor()
    ])
    input_image = Image.open(image_path) 
    input_image_torch = transform(input_image).to(device)

    x = input_image_torch
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    x = (x - torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)) / torch.Tensor(pixel_std).view(-1, 1, 1).to(device)
    return x[None, :, :, :]


def preprocess_biomedclip(preprocess, tokenizer, image_path, text, device="cuda"):
    """Preprocess image and text for Biomed CLIP."""
    bmc_img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    texts = tokenizer(text, context_length=256).to(device)
    return bmc_img, texts


def preprocess_groundingdino_img(image_path, device="cuda"):
    """Preprocess image for Grounding Dino."""
    _, image = load_image(image_path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((20, 20)),
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image.to(device)


class UnitTest:
    """Unit test for model"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_load_model(self):
        groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear = load_model(self.device)

    def test_load_preprocess(self):
        img_path = "./initial_experiments/toy_data/chest_x_ray.jpeg"
        text = "This is a image a 2 lungs."
        groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear = load_model(self.device)
        
        sam_img = preprocess_sam(sam, img_path, self.device)
        print("Sam image embedding shape:", sam_img.shape)
        
        bmc_img, bmc_txt = preprocess_biomedclip(preprocess_train, tokenizer, img_path, text, self.device)
        print("Biomed CLIP image embedding shape:", bmc_img.shape)
        print("Biomed CLIP text embedding shape:", bmc_txt.shape)
        
        groundingdino_img = preprocess_groundingdino_img(img_path, self.device)
        print("Grounding Dino image embedding shape:", groundingdino_img.shape)

    def test_embedding_generation(self):
        groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear = load_model(self.device)
        img_path = "./initial_experiments/toy_data/chest_x_ray.jpeg"
        text = "This is a image a 2 lungs."
        sam_img = preprocess_sam(sam, img_path, self.device)
        bmc_img, bmc_txt = preprocess_biomedclip(preprocess_train, tokenizer, img_path, text, self.device)
        groundingdino_img = preprocess_groundingdino_img(img_path, self.device)
        
        # SAM
        print(sam_img.shape)
        sam_img_embedding = sam.image_encoder(sam_img)
        print("SAM image embedding shape:", sam_img_embedding.shape)
        sam_img_embedding = sam_linear(sam_img_embedding)
        print("SAM image embedding shape:", sam_img_embedding.shape)
        
        # Biomed CLIP
        bmc_img_embedding = biomedclip.visual(bmc_img)
        bmc_txt_embedding = biomedclip.encode_text(bmc_txt)
        print("Biomed CLIP image embedding shape:", bmc_img_embedding.shape)
        print("Biomed CLIP text embedding shape:", bmc_txt_embedding.shape)
        
        # Grounding Dino
        groundingdino_img_embedding = groundingdino.backbone(groundingdino_img)
        groundingdino_txt_embedding = groundingdino.bert(bmc_txt_embedding)
        groundingdino_img_embedding = groundingdino_img_linear(groundingdino_img_embedding)
        groundingdino_txt_embedding = groundingdino_txt_linear(groundingdino_txt_embedding)
        print("Grounding Dino image embedding shape:", groundingdino_img_embedding.shape)
        print("Grounding Dino text embedding shape:", groundingdino_txt_embedding.shape)


if __name__ == "__main__":
    unit_test = UnitTest()
    # unit_test.test_load_model()
    # unit_test.test_load_preprocess()
    unit_test.test_embedding_generation()