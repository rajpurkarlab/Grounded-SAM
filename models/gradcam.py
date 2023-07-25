"""
This code was inspired by https://colab.research.google.com/github/kevinzakka/clip_playground/blob/main/CLIP_GradCAM_Visualization.ipynb. 
I adapted the CLIP Grad-CAM visualization code to the BioViL setting.

@software{zakka2021clipplayground,
    author = {Zakka, Kevin},
    month = {7},
    title = {{A Playground for CLIP-like Models}},
    url = {https://github.com/kevinzakka/clip_playground},
    version = {0.0.1},
    year = {2021}
}
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn

# from .CheXzero.zero_shot import load_chexzero

from .BioViL.text.utils import get_cxr_bert_inference as get_bert_inference
from .BioViL.image.utils import get_biovil_resnet_inference as get_image_inference

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize an array to [0, 1] range.

    Parameters:
        x (np.ndarray): array to normalize
    
    Returns:
        x (np.ndarray): normalized array
    """

    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def getAttMap(img, attn_map, blur=True):
    """
    Normalize and blur attention map.

    Parameters:
        img (np.ndarray): image
        attn_map (np.ndarray): attention map
        blur (bool): whether to blur the attention map
    
    Returns:
        attn_map (np.ndarray): normalized and blurred attention map
    """

    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    return attn_map

def viz_attn(img, attn_map, blur=True):
    """
    Visualize attention map.
    
    Parameters:
        img (np.ndarray): image
        attn_map (np.ndarray): attention map
        blur (bool): whether to blur the attention map
    """

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.show()
    
def load_image(img_path, resize=None):
    """
    Load image from path and resize if desired.

    Parameters:
        img_path (str): path to image
        resize (int): size to resize image to
    
    Returns:
        image (np.ndarray): (resized) image
    """
    
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad

def gradCAM(model, input, target, layer, input_size, gradcam_plus=True) -> torch.Tensor:
    """
    Compute Grad-CAM or Grad-CAM+ heatmap for a given input and target at a given layer.

    Parameters:
        model (nn.Module): model
        input (torch.Tensor): input
        target (torch.Tensor): target
        layer (nn.Module): layer to compute Grad-CAM at
        input_size (int): size of input
        gradcam_plus (bool): whether to compute Grad-CAM+ or not
    
    Returns:
        gradcam (torch.Tensor): Grad-CAM or Grad-CAM+ heatmap
    """
    
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.        
        output = model(input).projected_patch_embeddings
        output = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)), 1).squeeze(-1).squeeze(-1)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        if gradcam_plus:
            gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        tuple(input_size), #input.shape[2:]
        mode='bicubic',
        align_corners=False,
    )
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam

def get_gradcam_map_biovil(image_path, image_caption, input_size):
    """
    Compute Grad-CAM or Grad-CAM+ heatmap for a given image and caption.

    Parameters:
        image_path (str): path to image
        image_caption (str): caption for image
        input_size (int): size of input
        gradcam_plus (bool): whether to use Grad-CAM+ (clamp gradients) or not
        seg_targets (list): list of anatomical structures to compute Grad-CAM at
    
    Returns:
        map (np.ndarray): Grad-CAM or Grad-CAM+ heatmap
    """

    # Load BioViL image and text encoders
    text_inference = get_bert_inference()
    image_inference = get_image_inference()

    saliency_layer = "layer4" # We record gradients at layer 4, the last layer of the ResNet encoder before attention pooling

    origimg = np.array(Image.open(image_path).convert('L')) # Load original image

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preprocess image for phrase grounding
    image_input = image_inference.transform(Image.fromarray(origimg)).unsqueeze(0).to(device)
    image_np = load_image(image_path, input_size[0])

    attn_map = gradCAM( # Compute Grad-CAM or Grad-CAM+ heatmap
        image_inference.model.to(device),
        image_input.to(device),
        text_inference.get_embeddings_from_prompt(image_caption).float().to(device),
        getattr(image_inference.model.encoder.encoder, saliency_layer),
        input_size,
        gradcam_plus=True
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()
    map = getAttMap(image_np, attn_map, blur=True)
    
    return map