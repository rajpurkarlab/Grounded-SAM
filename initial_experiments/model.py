"""Model and preprocess function loading for grounded SAM experiment.
"""
import warnings
warnings.simplefilter("ignore")
import sys
sys.path.extend(["../", "./"])

import pdb
import torch
import torchvision
from torchvision.ops import box_convert
from PIL import Image
import cv2
from functools import partial
import open_clip
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry, build_sam, SamPredictor
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

from models.grounded_sam import *
from groundingdino.util.misc import nested_tensor_from_tensor_list
from models.GroundingDINO.groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map_nocate
from models.GroundingDINO.groundingdino.util.inference import load_model, preprocess_caption

from linear_probe import LinearProbe


class myGroundingDino:
    
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
        self.device = device
        
        # Load linear probe for Grounding Dino image embedding
        groundingdino_input_dims = [
            [1, 256, 100, 100],
            [1, 512, 50, 50],
            [1, 1024, 25, 25],
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
            [1, 256, 256]
        ]
        self.txt_linear = LinearProbe(
            groundingdino_txt_dims,
            512,
            device,
        )
        if txt_linear_ckpt:
            self.txt_linear.load_state_dict(torch.load(txt_linear_ckpt, map_location=device))


    def load_image(self, image_path: str):
        """Load image for Grounding Dino.
        
        Helper function for preprocess_img, lightly adopted from Grounding Dino codebase and removed the center crop.
        """
        transform = torchvision.transforms.Compose([
                # T.RandomResize([800], max_size=1333),
                # S.CenterCrop(800),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                torchvision.transforms.Resize((800, 800)),
        ])
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        image_transformed = transform(image_source)
        return image, image_transformed


    def preprocess_img(self, image_paths):
        """Preprocess image for Grounding Dino.
        
        Inputs:
            - image_paths: list of image paths
        """
        source_images, images = [], []
        for image_path in image_paths:
            source_image, image = self.load_image(image_path)
            source_images.append(source_image)
            images.append(image)

        # Convert tensor to nested tensor
        images = nested_tensor_from_tensor_list(images).to(self.device)
        return source_images, images.to(self.device)
    

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
    
    def predict(self, image_path: str, caption: str, box_threshold: float):  
        # Prepare images
        source_image, image = self.preprocess_img(image_path)
        caption = [preprocess_caption(c) for c in caption]

        # Get prediction
        outputs = self.model(image, captions=caption)
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

        # Filter results with confidence threshold
        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        # Filter for the highest confidence box
        mask = logits.max(dim=1)[0] == logits.max(dim=1)[0].max()
        logits = logits[mask]  # logits.shape = (1, 256)
        boxes = boxes[mask]  # boxes.shape = (1, 4)

        # Resize boxes from [0, 1] to original dimension of the image
        h, w, _ = source_image[0].shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").detach().numpy()

        return boxes, logits.max(dim=1)[0]


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


# class mySAM:

#     def __init__(
#         self,
#         model_name="vit_l",
#         ckpt_file="./initial_experiments/ckpts/sam_vit_l_0b3195.pth",
#         img_linear_ckpt=None,
#         device="cuda",
#     ):
#         # Load Grounded SAM
#         self.model = sam_model_registry[model_name](checkpoint=ckpt_file)
#         print(self.model.image_encoder.img_size)
#         raise NameError()
#         self.model.to(device)
#         self.device = device

#         # Load linear probe for SAM image embedding
#         sam_input_dims = [
#             [1, 256, 64, 64]
#         ]
#         self.img_linear = LinearProbe(
#             sam_input_dims, 
#             512,
#             device,
#         )
#         if img_linear_ckpt:
#             self.img_linear.load_state_dict(torch.load(img_linear_ckpt, map_location=device))


#     def preprocess_img(self, image_paths):
#         """Preprocess image for SAM.
        
#         Inputs:
#             - image_paths: list of image paths
#         """
        
#         images = []
#         for image_path in image_paths:
#             input_image = Image.open(image_path) 
#             images.append(input_image)

#         transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((20, 20)),
#             torchvision.transforms.ToTensor()
#         ])
#         input_image_torch = [transform(img).to(self.device) for img in images]
#         input_image_torch = torch.stack(input_image_torch)

#         x = input_image_torch
#         pixel_mean = [123.675, 116.28, 103.53]
#         pixel_std = [58.395, 57.12, 57.375]
#         x = (x - torch.Tensor(pixel_mean).view(-1, 1, 1).to(self.device)) / torch.Tensor(pixel_std).view(-1, 1, 1).to(self.device)
#         return x

    
#     def get_img_emb(self, image_path):
#         """Get image embedding for SAM"""
#         sam_img = self.preprocess_img(image_path)
#         sam_img_embedding = self.model.image_encoder(sam_img)
#         return sam_img_embedding


#     def align_img_emb(self, sam_img_embedding):
#         """Align image embedding to 512."""
#         sam_img_embedding = self.img_linear([sam_img_embedding])
#         return sam_img_embedding


#     def save_model(
#         self, 
#         ckpt_folder="./initial_experiments/ckpts/", 
#         backbone_file="sam.pth",
#         img_linear_ckpt="sam_img_linear.pth"
#     ):
#         """Save SAM backbone and linear probe for image embedding."""
#         torch.save(self.model.state_dict(), ckpt_folder + backbone_file)
#         torch.save(self.img_linear.state_dict(), ckpt_folder + img_linear_ckpt)


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

    
    def test_grounding_dino_predict(self):
        # Load model
        grounding_dino = myGroundingDino()
        IMAGE_PATH = ["./initial_experiments/toy_data/cat_dog.jpeg"]
        TEXT_PROMPT = ["2 dogs sitting in their kennels"]
        BOX_TRESHOLD = 0.35

        # Get predicted bbox
        boxes, logits = grounding_dino.predict(
            image_path=IMAGE_PATH,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
        )
        print(boxes, logits)
        print("Test grounding dino predict: SUCCESS!")


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
        bmc = myBiomedCLIP()
        
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


if __name__ == "__main__":
    unit_test = UnitTest()

    unit_test.test_grounding_dino()
    unit_test.test_grounding_dino_predict()
    # # unit_test.test_grounding_dino_save()

    # unit_test.test_biomed_clip()
    # # unit_test.test_biomed_clip_save()

    # unit_test.test_sam()
    # unit_test.test_sam_save()