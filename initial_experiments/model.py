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
from groundingdino.util.misc import nested_tensor_from_tensor_list
from models.GroundingDINO.groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map_nocate
from models.GroundingDINO.groundingdino.util.inference import load_image, load_model

from linear_probe import LinearProbe


class MyGroundingDino:
    
    def __init__(
        self,
        config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
        ckpt_file="./initial_experiments/ckpts/groundingdino_swint_ogc.pth",
        img_linear_ckpt=None,
        txt_linear_ckpt=None,
        device="cuda",
    ):
        """Grounding Dino - model(image encoder, text encoder), linear probe for image embedding and text embedding.
        """
        # Load grounding dino model
        self.model = load_model(config_file, ckpt_file)
        self.model.to(device)
        
        # Load linear probe for Grounding Dino image embedding
        groundingdino_input_dims = [
            [1, 256, 28, 28],
            [1, 512, 14, 14],
            [1, 1024, 7, 7],
        ]
        self.img_linear = LinearProbe(
            groundingdino_input_dims,
            512,
            device,
        )
        if img_linear_ckpt:
            self.img_linear.load_state_dict(torch.load(img_linear_ckpt, map_location=device))

        # Load linear probe for Grounding Dino text embedding
        groundingdino_txt_dims = [
            [1, 195, 256]
        ]
        self.txt_linear = LinearProbe(
            groundingdino_txt_dims,
            512,
            device,
        )
        if txt_linear_ckpt:
            self.txt_linear.load_state_dict(torch.load(txt_linear_ckpt, map_location=device))


    def preprocess_img(self, image_path, device="cuda"):
        """Preprocess image for Grounding Dino."""
        _, image = load_image(image_path)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
        ])
        image = transform(image)
        image = nested_tensor_from_tensor_list([image]).to(device)
        return image.to(device)
    

    def get_img_emb(self, image_path, device="cuda"):
        """Get image embedding for Grounding Dino."""
        # Preprocess
        img = self.preprocess_img(image_path, device)
        
        # Run backbone
        backbone_output, _ = self.model.backbone(img)
        groundingdino_img_embedding = []
        for emb in backbone_output:
            groundingdino_img_embedding.append(emb.tensors.to(device))
        return groundingdino_img_embedding
    

    def get_txt_emb(self, text, device="cuda"):
        """Get text embedding for Grounding Dino."""
        # Tokenize
        tokenized = self.model.tokenizer(text, padding="max_length", max_length=195, return_tensors="pt")
        for key, value in tokenized.items():
            tokenized[key] = value.to(device)

        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map_nocate(
            tokenized, self.model.specical_tokens, self.model.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.model.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.model.max_text_len, : self.model.max_text_len
            ]
            position_ids = position_ids[:, : self.model.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.model.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.model.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.model.max_text_len]

        if self.model.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        # Run text backbone
        bert_output = self.model.bert(**tokenized_for_encoder)
        groundingdino_txt_embedding = self.model.feat_map(bert_output["last_hidden_state"]).to(device)
        return groundingdino_txt_embedding

    
    def align_img_emb(self, groundingdino_img_embedding):
        """Align image embedding to 512."""
        groundingdino_img_embedding = self.img_linear(groundingdino_img_embedding)
        return groundingdino_img_embedding
    

    def align_txt_emb(self, groundingdino_txt_embedding):
        """Align text embedding to 512."""
        groundingdino_txt_embedding = self.txt_linear(groundingdino_txt_embedding)
        return groundingdino_txt_embedding
    

    def save_model(
            self, 
            ckpt_folder="./initial_experiments/ckpts/", 
            backbone_ckpt="groundingdino_backbone.pth",
            img_linear_ckpt="groundingdino_img_linear.pth", 
            txt_linear_ckpt="groundingdino_txt_linear.pth"
        ):
        """Save linear probe for image embedding and text embedding."""
        torch.save({"model": self.model.backbone.state_dict()}, ckpt_folder + backbone_ckpt)
        torch.save(self.img_linear.state_dict(), ckpt_folder + img_linear_ckpt)
        torch.save(self.txt_linear.state_dict(), ckpt_folder + txt_linear_ckpt)


class myBiomedCLIP:

    def __init__(
        self,
        config_file='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        ckpt_file=None,
        device="cuda",
    ):
        """Biomed CLIP - model(image encoder, text encoder)."""
        # Load Biomed CLIP
        biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(config_file)
        self.model = biomedclip.to(device)
        self.preprocess = preprocess_train
        self.tokenizer = open_clip.get_tokenizer(config_file)

        # Load checkpoint
        if ckpt_file:
            self.model.load_state_dict(torch.load(ckpt_file, map_location=device))

    
    def preprocess_img(self, image_path, device="cuda"):
        """Preprocess image for Biomed CLIP."""
        bmc_img = self.preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        return bmc_img
    

    def get_img_emb(self, image_path, device="cuda"):
        """Get image embedding for Biomed CLIP"""
        bmc_img = self.preprocess_img(image_path, device)
        bmc_img_embedding = self.model.visual(bmc_img)
        return bmc_img_embedding


    def get_txt_emb(self, text, device="cuda"):
        """Get text embedding for Biomed CLIP"""
        bmc_txt = self.tokenizer(text, context_length=256).to(device)
        bmc_txt_embedding = self.model.encode_text(bmc_txt)
        return bmc_txt_embedding

    
    def save_model(self, ckpt_folder="./initial_experiments/ckpts/", ckpt_file="biomedclip.pth"):
        """Save Biomed CLIP model."""
        torch.save(self.model.state_dict(), ckpt_folder + ckpt_file)


class mySAM:

    def __init__(
        self,
        ckpt_file="./initial_experiments/ckpts/sam_vit_l_0b3195.pth",
        device="cuda",
    ):
        # Load Grounded SAM
        self.model = sam_model_registry["vit_l"](checkpoint=ckpt_file)
        self.model.to(device)

    def preprocess_img(self, image_path, device="cuda"):
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

    
    def get_img_emb(self, image_path, device):
        sam_img = self.preprocess_img(image_path, device)
        sam_img_embedding = self.model.image_encoder(sam_img)
        return sam_img_embedding


    def save_model(self, ckpt_folder="./initial_experiments/ckpts/", ckpt_file="sam.pth"):
        """Save SAM model."""
        torch.save(self.model.state_dict(), ckpt_folder + ckpt_file)


class UnitTest:
    """Unit test for model.py.
    
    Also shows an example use case of the classes and functions in this file.
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_path = "./initial_experiments/toy_data/chest_x_ray.jpeg"
        self.text = "This is a image a 2 lungs."
    
    
    def test_grounding_dino(self):
        # Load model
        grounding_dino = MyGroundingDino()

        # Generate embedding
        groundingdino_img_embedding = grounding_dino.get_img_emb(self.img_path, self.device)
        print("Grounding Dino image embedding shape:")
        for emb in groundingdino_img_embedding:
            print(emb.shape)

        groundingdino_txt_embedding = grounding_dino.get_txt_emb(self.text, self.device)
        print("Grounding Dino text embedding shape:", groundingdino_txt_embedding.shape)

        groundingdino_img_embedding = grounding_dino.align_img_emb(groundingdino_img_embedding)
        print("After alignment, image embedding shape:", groundingdino_img_embedding.shape)

        groundingdino_txt_embedding = grounding_dino.align_txt_emb(groundingdino_txt_embedding)
        print("After alignment, text embedding shape:", groundingdino_txt_embedding.shape)
        print("Test grounding dino: SUCCESS!")
    

    def test_grounding_dino_save(self):
        # Load model
        grounding_dino = MyGroundingDino()

        # Save model
        grounding_dino.save_model()

        # Load model with saved checkpoint
        grounding_dino = MyGroundingDino(
            ckpt_file="./initial_experiments/ckpts/groundingdino_backbone.pth",
            img_linear_ckpt="./initial_experiments/ckpts/groundingdino_img_linear.pth", 
            txt_linear_ckpt="./initial_experiments/ckpts/groundingdino_txt_linear.pth"
        )
        print("Test grounding dino save: SUCCESS!")
    
    def test_sam(self):
        # Load model
        sam = mySAM()

        # Generate embedding
        sam_img_embedding = sam.get_img_emb(self.img_path, self.device)
        print("SAM image embedding shape:", sam_img_embedding.shape)
        print("Test sam: SUCCESS!")

    
    def test_sam_save(self):
        # Load model
        sam = mySAM()
        # Save model
        sam.save_model()
        # Load model with saved checkpoint
        sam = mySAM(ckpt_file="./initial_experiments/ckpts/sam.pth")
        print("Test sam save: SUCCESS!")


    def test_biomed_clip(self):
        # Load model
        bmc = myBiomedCLIP()
        
        # Generate embedding
        bmc_img_embedding = bmc.get_img_emb(self.img_path, self.device)
        print("Biomed CLIP image embedding shape:", bmc_img_embedding.shape)

        bmc_txt_embedding = bmc.get_txt_emb(self.text, self.device)
        print("Biomed CLIP text embedding shape:", bmc_txt_embedding.shape)
        print("Test biomed clip: SUCCESS!")
    

    def test_biomed_clip_save(self):
        # Load model
        bmc = myBiomedCLIP()
        # Save model
        bmc.save_model(ckpt_folder="./initial_experiments/ckpts/", ckpt_file="biomedclip.pth")
        # Load model with saved checkpoint
        bmc = myBiomedCLIP(ckpt_file="./initial_experiments/ckpts/biomedclip.pth")


if __name__ == "__main__":
    unit_test = UnitTest()
    
    unit_test.test_grounding_dino()
    unit_test.test_grounding_dino_save()

    unit_test.test_biomed_clip()
    unit_test.test_biomed_clip_save()

    unit_test.test_sam()
    unit_test.test_sam_save()