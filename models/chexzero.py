import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir + "/CheXzero/")

from zero_shot import load_clip
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, ToTensor


def load_chexzero_and_transform(ckpt="/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/models/CheXzero/checkpoints/best_64_5e-05_original_22000_0.864.pt"):
    # Load model
    model_path = ckpt
    model = load_clip(
        model_path=model_path,
        pretrained=True,
    )
    
    # Load transformation
    input_resolution = 224
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        ToTensor(),
        Normalize([101.48761, 101.48761, 101.48761], [83.43944, 83.43944, 83.43944]),
        Resize((input_resolution, input_resolution), interpolation=InterpolationMode.BICUBIC),
    ]
    transform = Compose(transformations)

    return model, transform


if __name__ == "__main__":
    model, transform = load_chexzero_and_transform()