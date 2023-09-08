from PIL import Image
import numpy as np
from pathlib import Path
import torch
import open_clip

from .gradcam import get_gradcam_map_biovil #, get_gradcam_map_bmclip, get_gradcam_map_chexzero

from .BioViL.text import get_cxr_bert_inference
from .BioViL.image import get_biovil_resnet_inference
from .BioViL.vlp import ImageTextInferenceEngine

# Load BioViL image and text encoders
text_inference = get_cxr_bert_inference()
image_inference = get_biovil_resnet_inference()

image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)

def load_biomed_clip(device):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.to(device)
    return model, tokenizer, preprocess_train, preprocess_val

def run_biomed_clip(img_path, text_prompt, gradcam=False):
    model, tokenizer, preprocess_train, preprocess_val = load_biomed_clip(device)
    if gradcam:
        pass
    else:
        # Using `model`, compute similarity grid between embeddings of `img_path` and `text_prompt`
        pass

def run_biovil(img_path, text_prompt, gradcam=False):
    if gradcam:
        return get_gradcam_map_biovil(img_path, text_prompt, np.array(Image.open(img_path).size))
    # print(img_path, text_prompt)
    
    similarity_map = image_text_inference.get_similarity_map_from_raw_data( # Call BioViL phrase grounding method (similarity matrix)
        image_path=Path(img_path),
        query_text=text_prompt,
        interpolation="bilinear",
    )
    return similarity_map

# def run_chexzero(img_path, text_prompt):