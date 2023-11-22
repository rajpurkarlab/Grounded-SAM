import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir + "/myBioViL/")

import pdb

import torch
import torch.optim as optim
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from biovil_model import ImageTextModel
from dataset_mscxr import create_chest_xray_transform_for_inference, remap_to_uint8

def load_biovil_and_transform(
    ckpt_path=None
):  
    # Load model
    text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
    image_inference = get_image_inference(ImageModelType.BIOVIL_T)
    model = ImageTextModel(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
        width=1024,
        height=1024,
    )

    # Load transformation
    transform = create_chest_xray_transform_for_inference()
    return model, transform

if __name__ == "__main__":
    model, transform = load_biovil_and_transform()