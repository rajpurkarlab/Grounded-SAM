import sys
import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

from . import clip
from .model import CLIP

def load_clip(model_path, pretrained=False, context_length=77): 
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    device = torch.device("cpu")
    if pretrained is False: 
        # use new model params
        params = {
            'embed_dim':768,
            'image_resolution': 320,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 16,
            'context_length': context_length, 
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12
        }

        model = CLIP(**params)
    else: 
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False) 
    try: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    except: 
        print("Argument error. Set pretrained = True.", sys.exc_info()[0])
        raise
    return model

def load_chexzero(
    model_path: str, 
    pretrained: bool = True, 
    context_length: bool = 77, 
):
    # load model
    model = load_clip(
        model_path=model_path, 
        pretrained=pretrained, 
        context_length=context_length
    )

    # load data
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    # if using CLIP pretrained model
    if pretrained: 
        # resize to input resolution of pretrained clip model
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    
    return model, transform