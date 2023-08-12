from models.grounded_sam import *
from PIL import Image
import cv2
import open_clip
from segment_anything.utils.transforms import ResizeLongestSide
import torchvision

from linear_probe import LinearProbe

from groundingdino.util.misc import (
    nested_tensor_from_tensor_list,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define preprocess functions
def preprocess_sam(sam, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_image = sam.preprocess(transformed_image)
    return input_image #.to(device)

def preprocess_biomedclip(preprocess, tokenizer, image_path, text):
    bmc_inp_img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    texts = tokenizer(text, context_length=256).to(device)
    return bmc_inp_img, texts

def preprocess_groundingdino_img(image_path):
    from models.GroundingDINO.groundingdino.util.inference import load_image
    _, image = load_image(image_path)
    return image

def load_models():
    # Load Grounding Dino
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename) # groundingdino.backbone, groundingdino.bert, groundingdino.tokenizer
    
    # Load Grounded SAM
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint) # sam.image_encoder, sam.prompt_encoder
    sam.to(device)

    # Load Biomed CLIP
    biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    biomedclip.to(device)
    
    return groundingdino, sam, biomedclip, tokenizer, preprocess_train, preprocess_val

def experiment():
    groundingdino, sam, biomedclip, tokenizer, preprocess_train, preprocess_val = load_models()
    
    # Try an example with image and texts
    image_path = "datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"
    caption = "Lung lesion"
   
    # Get embeddings
    samples = [preprocess_groundingdino_img(image_path)]
    if isinstance(samples, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(samples)
    groundingdino_img_emb, _ = groundingdino.backbone(samples)

    # groundingdino_txt_emb = groundingdino.bert(groundingdino.tokenizer(caption))
    
    with torch.no_grad():
        sam_img_emb = sam.image_encoder(preprocess_sam(sam, image_path))[0][0]
        img, txt = preprocess_biomedclip(preprocess_val, tokenizer, image_path, caption)
        biomedclip_img_emb, biomedclip_txt_emb, _ = biomedclip(img, txt)
    
    print("\n====== GROUNDING-DINO")
    print(len(groundingdino_img_emb))
    for emb in groundingdino_img_emb:
        # print(len(emb))
        print(f"\t{emb.shape}")
    # print(groundingdino_txt_emb.shape)
    print("\n====== SAM")
    print(sam_img_emb.shape)
    print("\n====== BIOMED-CLIP")
    print(biomedclip_img_emb.shape)
    print(biomedclip_txt_emb.shape)

"""
func(batch of images (assume an attribute is pathology), all encoders, linear layers):
    compute all embeddings for every img in batch, apply linear layers as needed -> output: bmi, bmt, si, gdi, gdt - dim: (batch_size, embedding_shape*) *same for all encoders
    1) e.g. SAM
    negative_mode='paired'
    query: si
    positive_key: bmi
    negative_keys: for i in bmi: negative_keys = filter bmi to only include images of different pathology than i
"""

def compute_loss(batch, pathologies, groundingdino, sam, biomedclip, tokenizer, preprocess_train):
    """
    batch: list of image paths
    pathologies: list of pathologies
    """
    for i, image_path in enumerate(batch):
        samples = [preprocess_groundingdino_img(image_path)]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        groundingdino_img_emb, _ = groundingdino.backbone(samples)
        
        bmi = []
        bmt = []
        si = []
        gdi = []
        gdt = []
        
        with torch.no_grad():
            sam_img_emb = sam.image_encoder(preprocess_sam(sam, image_path))[0][0]
            img, txt = preprocess_biomedclip(preprocess_train, tokenizer, image_path, pathologies[i])
            biomedclip_img_emb, biomedclip_txt_emb, _ = biomedclip(img, txt)
            bmi.append(biomedclip_img_emb.to(device))
            bmt.append(biomedclip_txt_emb.to(device))
        
        gd_img_emb = []
        for emb in groundingdino_img_emb:
            gd_img_emb.append(emb.tensors.to(device))

        grounding_dino_linear = LinearProbe(gd_img_emb, biomedclip_img_emb.shape[1], device)
        grounding_dino_emb_aligned = grounding_dino_linear(gd_img_emb)
        gdi.append(grounding_dino_emb_aligned)
        
        si_emb = [sam_img_emb[None, :].to(device)]
        sam_linear = LinearProbe(si_emb, biomedclip_img_emb.shape[1], device)
        sam_emb_aligned = sam_linear(si_emb)
        si.append(sam_emb_aligned)
        
        # TODO: Grounding dino text
    
    bmi = torch.stack(bmi)
    bmt = torch.stack(bmt)
    si = torch.stack(si)
    gdi1 = torch.stack(gdi1)
    gdi2 = torch.stack(gdi2)
    gdi3 = torch.stack(gdi3)
    gdt = torch.stack(gdt)
    
    print(bmi.shape, bmt.shape, si.shape, gdi1.shape, gdi2.shape, gdi3.shape, gdt.shape)
    
    loss_sam = 0
    loss_groundingdino_img = 0
    loss_groundingdino_txt = 0
    return loss_sam + loss_groundingdino_img + loss_groundingdino_txt
          
if __name__ == "__main__":
    groundingdino, sam, biomedclip, tokenizer, preprocess_train, preprocess_val = load_models()
    compute_loss(["datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"],
                 "Lung lesion",
                 groundingdino,
                 sam,
                 biomedclip,
                 tokenizer,
                 preprocess_train)