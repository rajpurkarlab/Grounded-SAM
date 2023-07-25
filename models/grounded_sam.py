import os, sys
import numpy as np
import torch
import requests
from PIL import Image

def env_setup():
    sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

    # If you have multiple GPUs, you can set the GPU to use here.
    # The default is to use the first GPU, which is usually GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

from segment_anything import build_sam, SamPredictor

from huggingface_hub import hf_hub_download

"""# Load Grounding DINO model"""
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def load_models():
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    return groundingdino_model, sam_predictor

def run_grounded_sam(image_path, text_prompt, groundingdino_model, sam_predictor, BOX_TRESHOLD=0.3, TEXT_TRESHOLD=0.25, device='cuda'):    
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    sam_predictor.set_image(image_source)

    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    
    image_mask = masks[0][0].cpu().numpy()
    
    return image_mask