import warnings
warnings.simplefilter("ignore")

from models.grounded_sam import *
from PIL import Image
import cv2
import open_clip
from segment_anything.utils.transforms import ResizeLongestSide
import torchvision
from info_nce import InfoNCE

from linear_probe import LinearProbe

from models.GroundingDINO.groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map_nocate
)
from groundingdino.util.misc import (
    nested_tensor_from_tensor_list,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define preprocess functions
def preprocess_sam(sam, image_path):
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # transform = ResizeLongestSide(sam.image_encoder.img_size)
    # input_image = transform.apply_image(image)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((20,20)),
        torchvision.transforms.ToTensor()
    ])
    input_image = Image.open(image_path) 
    input_image_torch = transform(input_image).to(device)
    # torch.as_tensor(input_image, device=device)
    # transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    x = input_image_torch
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    x = (x - torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)) / torch.Tensor(pixel_std).view(-1, 1, 1).to(device)
    print(x.shape)

    # input_image = sam.preprocess(input_image_torch[None, :, :, :])
    # print(input_image.shape)
    return x[None, :, :, :] #.to(device)

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

def compute_loss(batch, pathologies, groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear):
    """
    batch: list of image paths
    pathologies: list of pathologies
    """
    bmi = []
    bmt = []
    si = []
    gdi = []
    gdt = []

    tokenized = groundingdino.tokenizer(pathologies, padding="max_length", max_length=195, return_tensors="pt")
    (
        text_self_attention_masks,
        position_ids
    ) = generate_masks_with_special_tokens_and_transfer_map_nocate(
        tokenized, groundingdino.specical_tokens, groundingdino.tokenizer
    )

    if text_self_attention_masks.shape[1] > groundingdino.max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : groundingdino.max_text_len, : groundingdino.max_text_len
        ]
        position_ids = position_ids[:, : groundingdino.max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : groundingdino.max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : groundingdino.max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : groundingdino.max_text_len]

    if groundingdino.sub_sentence_present:
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids
    else:
        tokenized_for_encoder = tokenized

    bert_output = groundingdino.bert(**tokenized_for_encoder)  # bs, 195, 768
    groundingdino_txt_emb = groundingdino.feat_map(bert_output["last_hidden_state"]).to(device)  # bs, 195, d_model
    
    for i in range(len(batch)):
        emb = groundingdino_txt_emb[i, :][None, :]
        gdt.append(groundingdino_txt_linear(emb).squeeze())
     
    for i, image_path in enumerate(batch):
        samples = [preprocess_groundingdino_img(image_path)]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        groundingdino_img_emb, _ = groundingdino.backbone(samples)
        
        # TODO: need to bring sam outside of torch.no_grad but may run into GPU issue.
        sam_img_emb = sam.image_encoder(preprocess_sam(sam, image_path))[0][0]
        
        with torch.no_grad():
            img, txt = preprocess_biomedclip(preprocess_train, tokenizer, image_path, pathologies[i])
            biomedclip_img_emb, biomedclip_txt_emb, _ = biomedclip(img, txt)
            bmi.append(biomedclip_img_emb.to(device).squeeze())
            bmt.append(biomedclip_txt_emb.to(device).squeeze())
        
        gd_img_emb = []
        for emb in groundingdino_img_emb:
            gd_img_emb.append(emb.tensors.to(device))

        grounding_dino_emb_aligned = groundingdino_img_linear(gd_img_emb)
        gdi.append(grounding_dino_emb_aligned.squeeze())
        
        si_emb = [sam_img_emb[None, :].to(device)]
        sam_emb_aligned = sam_linear(si_emb)
        si.append(sam_emb_aligned.squeeze())
            
    bmi = torch.stack(bmi)
    bmt = torch.stack(bmt)
    si = torch.stack(si)
    gdi = torch.stack(gdi)
    gdt = torch.stack(gdt)
            
    path2list = {}
    path2list_t = {}
    bmif = []
    bmtf = []
    uniq = list(set(pathologies))
    for path in uniq:
        l = []
        t = []
        for i, p in enumerate(pathologies):
            if p == path:
                l.append(bmi[i])
                t.append(bmt[i])
        path2list[path] = l
        path2list_t[path] = t
        
    for i, path in enumerate(pathologies):
        l = []
        t = []
        for p in uniq:
            if p != path:
                l += path2list[p]
                t += path2list_t[p]
        bmif.append(torch.stack(l))
        bmtf.append(torch.stack(t))
    
    bmif = torch.stack(bmif)
    bmtf = torch.stack(bmtf)
    
    loss = InfoNCE(negative_mode='paired')
    loss_sam = loss(si, bmi, bmif)
    loss_groundingdino_img = loss(gdi, bmi, bmif)
    loss_groundingdino_txt = loss(gdt, bmt, bmtf)
    
    return loss_sam + loss_groundingdino_img + loss_groundingdino_txt
          
if __name__ == "__main__":
    groundingdino, sam, biomedclip, tokenizer, preprocess_train, preprocess_val = load_models()
    
    grounding_dino_input_dims = [
        [1, 256, 100, 100],
        [1, 512, 50, 50],
        [1, 1024, 25, 25],
    ]
    grounding_dino_linear = LinearProbe(
        grounding_dino_input_dims,
        512,
        device,
        )
    
    sam_input_dims = [
        [1, 256, 64, 64]
    ]
    sam_linear = LinearProbe(
        sam_input_dims, 
        512,
        device,
        )
    
    groundingdino_txt_dims = [
        [1, 195, 256]
    ]
    grounding_dino_linear_txt = LinearProbe(
        groundingdino_txt_dims,
        512,
        device,
        )
    
    loss = compute_loss(
                ["datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg", "datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"],
                ["Lung lesion", "Cardiomegaly"],
                # ["datasets/chexlocalize/CheXpert/test/patient64741/study1/view1_frontal.jpg"],
                # ["Lung lesion"],
                groundingdino,
                sam,
                biomedclip,
                tokenizer,
                preprocess_train,
                grounding_dino_linear,
                grounding_dino_linear_txt,
                sam_linear)
    print(loss)