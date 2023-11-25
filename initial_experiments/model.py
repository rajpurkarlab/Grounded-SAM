"""Model and preprocess function loading for grounded SAM experiment.
"""
import warnings
warnings.simplefilter("ignore")
import sys
sys.path.extend(["../", "./"])

import pdb
import time
import torch
import torchvision
from torchvision.ops import box_convert, box_iou
from PIL import Image
import cv2
from functools import partial
import open_clip
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry, build_sam, SamPredictor
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
import bisect

# Grounded sam
from models.grounded_sam import *
from groundingdino.util.misc import nested_tensor_from_tensor_list
from models.GroundingDINO.groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map_nocate
from models.GroundingDINO.groundingdino.util.inference import load_model, preprocess_caption
from groundingdino.util.utils import get_phrases_from_posmap

# Load CheXzero
from models.chexzero import load_chexzero_and_transform
import models.CheXzero.clip as clip
from models.biovil import load_biovil_and_transform, remap_to_uint8

from linear_probe import LinearProbe
from utils import explore_tensor
from dataset_mimic import load_data as load_mimic
from dataset_pascal import load_data as load_pascal



class myGroundingDino:
    
    def __init__(
        self,
        d=128,
        config_file="./initial_experiments/ckpts/GroundingDINO_SwinT_OGC.py",
        ckpt_file="./initial_experiments/ckpts/groundingdino_swint_ogc.pth",
        # ckpt_file="./initial_experiments/ckpts/groundingdino_backbone_5151.pth",
        img_linear_ckpt=None,
        txt_linear_ckpt=None,
        device="cuda",
    ):
        """Grounding Dino - model(image encoder, text encoder), linear probe for image embedding and text embedding.
        """
        # Load grounding dino model
        self.model = load_model(config_file, ckpt_file)
        self.model.to(device)
        self.device = device
        
        # Load linear probe for Grounding Dino image embedding
        groundingdino_input_dims = [
            [1, 256, 32, 32],
            [1, 512, 16, 16],
            [1, 1024, 8, 8],
        ]
        self.img_linear = LinearProbe(
            groundingdino_input_dims,
            d,
            device,
        )
        if img_linear_ckpt:
            self.img_linear.load_state_dict(torch.load(img_linear_ckpt, map_location=device))

        # Load linear probe for Grounding Dino text embedding
        groundingdino_txt_dims = [
            [1, 256, 256]
        ]
        self.txt_linear = LinearProbe(
            groundingdino_txt_dims,
            d,
            device,
        )
        if txt_linear_ckpt:
            self.txt_linear.load_state_dict(torch.load(txt_linear_ckpt, map_location=device))


    def preprocess_img(self, images):
        """Preprocess image for Grounding Dino.
        
        Inputs:
            - images: tensor of shape (B, C, H, W)
        """
        # Convert tensor to nested tensor
        images = nested_tensor_from_tensor_list(images).to(self.device)
        return None, images.to(self.device)
    

    def get_img_emb(self, image_path):
        """Get image embedding for Grounding Dino."""
        # Preprocess
        _, img = self.preprocess_img(image_path)

        # Run backbone
        backbone_output, _ = self.model.backbone(img)
        groundingdino_img_embedding = []
        for emb in backbone_output:
            groundingdino_img_embedding.append(emb.tensors.to(self.device))
        return groundingdino_img_embedding
    

    def get_txt_emb(self, text):
        """Get text embedding for Grounding Dino."""
        # Tokenize
        tokenized = self.model.tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        for key, value in tokenized.items():
            tokenized[key] = value.to(self.device)

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
        groundingdino_txt_embedding = self.model.feat_map(bert_output["last_hidden_state"]).to(self.device)
        return groundingdino_txt_embedding

    
    def align_img_emb(self, groundingdino_img_embedding):
        """Align image embedding to 512."""
        groundingdino_img_embedding = self.img_linear(groundingdino_img_embedding)
        return groundingdino_img_embedding
    

    def align_txt_emb(self, groundingdino_txt_embedding):
        """Align text embedding to 512."""
        groundingdino_txt_embedding = self.txt_linear([groundingdino_txt_embedding])
        return groundingdino_txt_embedding


    def inference(
        self,
        image,
        caption,
        original_img_size,
        box_threshold,
        text_threshold,
    ):
        """Open-vocabulary phrase grounding using Grounding Dino during inference.

        Args:
            - image: tensor of shape (B, C, H, W).
            - caption: list of string captions.
            - original_img_size: list of tuple of (H, W) of original image size.
            - box_threshold: float, threshold for box confidence score.
            - text_threshold: float, threshold for text confidence score.

        Returns:
            - boxes: list of predicted bounding boxes.
                - boxes.length = image.length
                - boxes[i] has shape (n_i, 4).
                - n_i is different for each boxes[i] due to filtering.
            - logits: list of predicted text confidence scores.
            - phrases: list of predicted phrases.
        """
        B = len(image)

        # Prepare images and text
        caption = [preprocess_caption(c) for c in caption]

        # Get prediction
        outputs = self.model(image, captions=caption)
        prediction_logits = outputs["pred_logits"].sigmoid()  # prediction_logits.shape = (B, nq, 256), where nq = 900
        prediction_boxes = outputs["pred_boxes"]  # prediction_boxes.shape = (B, nq, 4)
        
        # Resize boxes from [0, 1] to original dimension of the image
        for i in range(B):
            h, w = original_img_size[i]
            prediction_boxes[i] = prediction_boxes[i] * torch.Tensor([w, h, w, h]).to(self.device)
        prediction_boxes = box_convert(boxes=prediction_boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # Filter box based on threshold
        boxes, logits = [], []
        for b in range(B):
            mask = prediction_logits[b].max(dim=1)[0] > box_threshold
            boxes.append(prediction_boxes[b][mask])
            logits.append(prediction_logits[b][mask])
        
        # Tokenize caption
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        # Separate tokenized into single samples
        tokenized_samples = [{} for _ in range(B)]
        for key, val in tokenized.items():
            for b in range(B):
                tokenized_samples[b][key] = val[b]

        # Copied from GD predict codebase, return a list (image-level) of list (object-level) of phrases
        PHRASES = []
        for b in range(B):
            sep_idx = [i for i in range(len(tokenized_samples[b]['input_ids'])) if tokenized_samples[b]['input_ids'][i] in [101, 102, 1012]]
            
            phrases = []
            for logit in logits[b]:
                max_idx = logit.argmax()
                insert_idx = bisect.bisect_left(sep_idx, max_idx)
                right_idx = sep_idx[insert_idx]
                left_idx = sep_idx[insert_idx - 1]
                phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized_samples[b], tokenizer, left_idx, right_idx).replace('.', ''))
            PHRASES.append(phrases)

        # Filter for the highest confidence box for each phrase
        new_boxes = []
        new_logits = []
        for b in range(B):
            # Remove empty phrases
            phrases = [string for index, string in enumerate(PHRASES[b]) if string != ""]
            
            bbs = [] # max logit box - shape: (num_phrases, 4)
            lls = [] # the logit for the max logit box - shape: (num_phrases, 256)
            # Get the max-logit box for each phrase in the caption
            for text_prompt in caption[b].split("."):
                text_prompt = text_prompt.strip()
                if text_prompt:
                    f = [index for index, string in enumerate(PHRASES[b]) if string in text_prompt]
                    try:
                        mask = logits[b][f].max(dim=1)[0] == logits[b][f].max(dim=1)[0].max(dim=0)[0] # mask.shape = (B, n)
                        bbs.append(boxes[b][f][mask])
                        lls.append(logits[b][f][mask])
                    except: # when no box corresponding to a certain phrase
                        bbs.append(torch.zeros((1,4), device='cuda'))
                        lls.append(torch.zeros((1,256), device='cuda'))
            new_boxes.append(torch.cat(bbs, dim=0))
            new_logits.append(torch.cat(lls, dim=0))     
        
        # During evaluation, we only have one image
        if B == 1: 
            boxes = torch.stack(new_boxes)
            logits = torch.stack(new_logits)
            logits = torch.stack([logit.max(dim=1)[0] for logit in logits])
            return boxes, logits

        return new_boxes, [logit.max(dim=1)[0] for logit in new_logits]
        

    def save_model(
            self, 
            ckpt_folder="./initial_experiments/ckpts/", 
            backbone_ckpt="groundingdino_backbone.pth",
            img_linear_ckpt="groundingdino_img_linear.pth", 
            txt_linear_ckpt="groundingdino_txt_linear.pth"
        ):
        """Save linear probe for image embedding and text embedding."""
        torch.save({"model": self.model.state_dict()}, ckpt_folder + backbone_ckpt) # self.model.backbone
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
        self.device=device

        # Load checkpoint
        if ckpt_file:
            self.model.load_state_dict(torch.load(ckpt_file, map_location=device))

    
    def preprocess_img(self, image_paths):
        """Preprocess image for Biomed CLIP.
        
        Inputs:
            - image_paths: list of image paths
        """
        images = []
        for image_path in image_paths:
            bmc_img = self.preprocess(Image.open(image_path)).to(self.device)
            images.append(bmc_img)
        images = torch.stack(images)
        return images
    

    def get_img_emb(self, image_path):
        """Get image embedding for Biomed CLIP"""
        bmc_img = self.preprocess_img(image_path)
        bmc_img_embedding = self.model.visual(bmc_img)
        return bmc_img_embedding


    def get_txt_emb(self, text):
        """Get text embedding for Biomed CLIP"""
        bmc_txt = self.tokenizer(text, context_length=256).to(self.device)
        bmc_txt_embedding = self.model.encode_text(bmc_txt)
        return bmc_txt_embedding

    
    def save_model(self, ckpt_folder="./initial_experiments/ckpts/", ckpt_file="biomedclip.pth"):
        """Save Biomed CLIP model."""
        torch.save(self.model.state_dict(), ckpt_folder + ckpt_file)


class mySAM:

    def __init__(
        self,
        model_name="vit_l",
        ckpt_file="./initial_experiments/ckpts/sam_vit_l_0b3195.pth",
        img_linear_ckpt=None,
        device="cuda",
    ):
        # Load Grounded SAM
        # self.model = sam_model_registry[model_name](checkpoint=ckpt_file)
        self.model = self._build_sam(
            checkpoint=ckpt_file,
            image_size=1024
        )
        self.model.to(device)
        self.device = device

        # Load linear probe for SAM image embedding
        sam_input_dims = [
            [1, 256, 64, 64]
        ]
        self.img_linear = LinearProbe(
            sam_input_dims, 
            512,
            device,
        )
        if img_linear_ckpt:
            self.img_linear.load_state_dict(torch.load(img_linear_ckpt, map_location=device))

    def _build_sam(
        self,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=None,
        image_size = 1024,
    ):
        prompt_embed_dim = 256
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        sam = Sam(
            image_encoder=ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
        sam.eval()
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            sam.load_state_dict(state_dict)
        return sam


    def preprocess_img(self, image_paths):
        """Preprocess image for SAM.
        
        Inputs:
            - image_paths: list of image paths
        """
        images = []
        transform = ResizeLongestSide(self.model.image_encoder.img_size)

        for image_path in image_paths:
            image = Image.open(image_path) 
            image = transform.apply_image(np.array(image))
            image = torch.as_tensor(image, device=self.device)
            image = image.permute(2, 0, 1).contiguous()[None, :, :, :]
            image = self.model.preprocess(image)
            images.append(image)
        
        images = torch.cat(images, dim=0)
        return images

    
    def get_img_emb(self, image_path):
        """Get image embedding for SAM"""
        sam_img = self.preprocess_img(image_path)
        print(sam_img.shape)
        sam_img_embedding, interm_features = self.model.image_encoder(sam_img)
        return sam_img_embedding


    def align_img_emb(self, sam_img_embedding):
        """Align image embedding to 512."""
        sam_img_embedding = self.img_linear([sam_img_embedding])
        return sam_img_embedding


    def save_model(
        self, 
        ckpt_folder="./initial_experiments/ckpts/", 
        backbone_file="sam.pth",
        img_linear_ckpt="sam_img_linear.pth"
    ):
        """Save SAM backbone and linear probe for image embedding."""
        torch.save(self.model.state_dict(), ckpt_folder + backbone_file)
        torch.save(self.img_linear.state_dict(), ckpt_folder + img_linear_ckpt)


class myCheXzero:
    
    def __init__(
        self,
        ckpt_file=None,
        device="cuda",
    ):
        """CheXZero - model(image encoder, text encoder).
        """
        # Load CheXZero
        self.model, self.preprocess_image = load_chexzero_and_transform()
        self.model.to(device)
        self.device = device

        # Load checkpoint
        if ckpt_file:
            self.model.load_state_dict(torch.load(ckpt_file, map_location=device))

    
    def preprocess_img(self, image_paths, desired_size=224):
        # Preprocess images
        images = []
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            # convert to PIL Image object
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            # preprocess
            # img = preprocess(img_pil, desired_size=resolution)  
            
            old_size = img.size
            # prcint(old_size)
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            img = img.resize(new_size, Image.LANCZOS)
            # create a new image and paste the resized on it

            new_img = Image.new('L', (desired_size, desired_size))
            new_img.paste(img, ((desired_size-new_size[0])//2,
                                (desired_size-new_size[1])//2))
            
            img = new_img
            
            img = img.convert("RGB")
            # img = self.preprocess_image(img).to(self.device)
            img = torchvision.transforms.ToTensor()(img).to(self.device)
            images.append(img)
        images = torch.stack(images)
        return images

    
    def preprocess_txt(self, caption):
        return clip.tokenize(caption, context_length=77).to(self.device)
    
    
    def get_img_emb(self, image_paths):
        """Get image embedding for CheXZero."""
        images = self.preprocess_img(image_paths)
        img_embedding = self.model.encode_image(images)
        return img_embedding
    

    def get_txt_emb(self, caption):
        """Get text embedding for CheXZero."""
        caption = self.preprocess_txt(caption)
        txt_embedding = self.model.encode_text(caption)
        # txt_embedding = txt_embedding / txt_embedding.norm(dim=-1, keepdim=True)
        return txt_embedding

    
    def predict(self, image_paths, caption):
        # Preprocess
        images = self.preprocess_img(image_paths)
        caption = self.preprocess_txt(caption)        

        # Run model
        logits_per_image, logits_per_text = self.model(images, caption)
        return logits_per_image, logits_per_text


class myBioViL:
    
    def __init__(
        self,
        ckpt_file=None,
        device="cuda",
    ):
        """BioViL - model(image encoder, text encoder).
        """
        # Load CheXZero
        self.model, self.transform = load_biovil_and_transform(ckpt_file)
        self.model.to(device)
        self.device = device

        # Load checkpoint
        if ckpt_file:
            self.model.load_state_dict(torch.load(ckpt_file, map_location=device))

    
    def get_local_img_emb(self, images):
        """Get localized image embedding for BioViL."""
        img_embedding = self.model.image_inference_engine.get_patchwise_projected_embeddings(
            images, normalize=True
        )
        return img_embedding
    
    
    def get_img_emb(self, images):
        """Get global image embedding for BioViL."""
        img_embedding = self.get_local_img_emb(images)
        img_embedding = torch.mean(img_embedding, dim=(1,2))
        return img_embedding
    

    def get_txt_emb(self, texts):
        """Get text embedding for BioViL."""
        max_length = 500
        texts = [text[:max_length] for text in texts]
        txt_embedding = self.model.text_inference_engine.get_embeddings_from_prompt(texts)
        return txt_embedding


class UnitTest:
    """Unit test for model.py.
    
    Also shows an example use case of the classes and functions in this file.
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_path = [
            "./initial_experiments/toy_data/chest_x_ray.jpeg",
            "./initial_experiments/toy_data/chest_x_ray_2.jpeg"
        ]
        self.text = [
            "This is a image a 2 lungs.",
            "This patient has symptom of pneumonia.",
        ]

        # Load dataloader - new
        h5_file_pascal = "/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/initial_experiments/data/pascal.h5"
        self.pascal_dataloader = load_pascal(batch_size=16, h5_file=h5_file_pascal, num_workers=4)

    def get_prompt(self, labels):
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
    
    def test_grounding_dino(self):
        # Load model
        grounding_dino = myGroundingDino()

        # Generate embedding
        groundingdino_img_embedding = grounding_dino.get_img_emb(self.img_path)
        print("Grounding Dino image embedding shape:")
        for emb in groundingdino_img_embedding:
            print(emb.shape)

        groundingdino_txt_embedding = grounding_dino.get_txt_emb(self.text)
        print("Grounding Dino text embedding shape:", groundingdino_txt_embedding.shape)

        groundingdino_img_embedding = grounding_dino.align_img_emb(groundingdino_img_embedding)
        print("After alignment, image embedding shape:", groundingdino_img_embedding.shape)

        groundingdino_txt_embedding = grounding_dino.align_txt_emb(groundingdino_txt_embedding)
        print("After alignment, text embedding shape:", groundingdino_txt_embedding.shape)
        print("Test grounding dino: SUCCESS!")

    
    def test_grounding_dino_inference(self):
        # Load model
        grounding_dino = myGroundingDino()
        BOX_TRESHOLD = 0.05
        TEXT_TRESHOLD = 0.05

        data = next(iter(self.pascal_dataloader))
        images = data["image_gd"].to(self.device)
        original_img_size = data["original_img_size"]
        prompts = [self.get_prompt(item) for item in data["labels"]]

        # Get predicted bbox
        boxes, logits = grounding_dino.inference(
            image=images,
            caption=prompts,
            original_img_size=original_img_size,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )
        print("Test grounding dino predict: SUCCESS!")
        pdb.set_trace()


    def test_grounding_dino_forward(self):
        # Load model
        grounding_dino = myGroundingDino()
        IMAGE_PATH = ["./initial_experiments/toy_data/cat_dog.jpeg", "./initial_experiments/toy_data/chest_x_ray.jpeg"]
        TEXT_PROMPT = ["2 dogs . kennels .", "2 lungs ."]
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        # Get predicted bbox
        boxes, logits = grounding_dino.forward(
            image_path=IMAGE_PATH,
            caption=TEXT_PROMPT,
        )
        print(boxes, logits)
        print("Test grounding dino forward: SUCCESS!")


    def test_grounding_dino_save(self):
        # Load model
        grounding_dino = myGroundingDino()

        # Save model
        grounding_dino.save_model()

        # Load model with saved checkpoint
        grounding_dino = myGroundingDino(
            ckpt_file="./initial_experiments/ckpts/groundingdino_backbone.pth",
            img_linear_ckpt="./initial_experiments/ckpts/groundingdino_img_linear.pth", 
            txt_linear_ckpt="./initial_experiments/ckpts/groundingdino_txt_linear.pth"
        )
        print("Test grounding dino save: SUCCESS!")

    
    def test_sam(self):
        # Load model
        sam = mySAM()

        # Generate embedding
        sam_img_embedding = sam.get_img_emb(self.img_path)
        print("SAM image embedding shape:", sam_img_embedding.shape)
        sam_img_embedding = sam.align_img_emb(sam_img_embedding)
        print("After alignment, image embedding shape:", sam_img_embedding.shape)
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
        bmc = myBiomedCLIP(device=self.device)
        
        # Generate embedding
        bmc_img_embedding = bmc.get_img_emb(self.img_path)
        print("Biomed CLIP image embedding shape:", bmc_img_embedding.shape)

        bmc_txt_embedding = bmc.get_txt_emb(self.text)
        print("Biomed CLIP text embedding shape:", bmc_txt_embedding.shape)
        print("Test biomed clip: SUCCESS!")
    

    def test_biomed_clip_save(self):
        # Load model
        bmc = myBiomedCLIP()
        # Save model
        bmc.save_model(ckpt_folder="./initial_experiments/ckpts/", ckpt_file="biomedclip.pth")
        # Load model with saved checkpoint
        bmc = myBiomedCLIP(ckpt_file="./initial_experiments/ckpts/biomedclip.pth")


    def test_chexzero_predict(self):
        # Load model
        chexzero = myCheXzero()

        # Generate embedding
        logits_per_image, logits_per_text = chexzero.predict(self.img_path, self.text)
        print(logits_per_image.shape, logits_per_text.shape)
        print("Test chexzero predict: SUCCESS!")


    def test_biovil(self):
        # Load model
        biovil = myBioViL()

        # Generate embedding
        img_embedding = biovil.get_img_emb(self.img_path)
        txt_embedding = biovil.get_txt_emb(self.text)
        print(img_embedding.shape, txt_embedding.shape)
        print("Test chexzero predict: SUCCESS!")


if __name__ == "__main__":
    unit_test = UnitTest()
    
    # Grounding DINO
    # unit_test.test_grounding_dino()
    # unit_test.test_grounding_dino_forward()
    unit_test.test_grounding_dino_inference()
    # unit_test.test_grounding_dino_save()

    # # Biomed CLIP
    # unit_test.test_biomed_clip()
    # unit_test.test_biomed_clip_save()

    # # SAM
    # unit_test.test_sam()
    # unit_test.test_sam_save()

    # # CheXZero
    # unit_test.test_chexzero_predict()

    # # BioViL
    # unit_test.test_biovil()