import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from linear_probe import LinearProbe
from adaptation_loss import load_models, compute_loss
import torch
from torchvision import transforms
from PIL import Image

class MIMICCXRDataset(Dataset):
    """MIMIC-CXR dataset."""

    def __init__(self, csv_file='/n/data1/hms/dbmi/rajpurkar/lab/CXR-ReDonE/data/mimic_train_impressions.csv', img_dir='/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files', size=(224,224), tensor=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.dataframe.dropna(subset=['report'], inplace=True)
        
        self.img_dir = img_dir
        self.size = size
        self.tensor = tensor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        dicom_id = row['dicom_id']
        study_id = row['study_id']
        subject_id = row['subject_id']
        report = row['report']
        
        image_path = f'{self.img_dir}/p{str(subject_id)[0:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        
        if self.tensor:
            img = Image.open(image_path)
            # Resize image to self.size
            img = img.resize(self.size)
            convert_tensor = transforms.ToTensor()
            sample = {'image': convert_tensor(img), 'image_path': image_path, 'report': report}
        else:
            sample = {'image_path': image_path, 'report': report}
        
        return sample

def load_data(batch_size=16, tensor=False):
    """Get dataloader for training.
    """
    dataset = MIMICCXRDataset(tensor=tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_model():
    """Return image encoder, text encoder, and linear probes."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    groundingdino, sam, biomedclip, tokenizer, preprocess_train, preprocess_val = load_models()
    
    groundingdino_input_dims = [
        [1, 256, 100, 100],
        [1, 512, 50, 50],
        [1, 1024, 25, 25],
    ]
    groundingdino_linear = LinearProbe(
        groundingdino_input_dims,
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
    groundingdino_linear_txt = LinearProbe(
        groundingdino_txt_dims,
        512,
        device,
        )

    return groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_linear, groundingdino_linear_txt, sam_linear
    # NOTE: Linear probes still need to be tuned

def train(hyparams, output_path, model_paths):
    """Train the model."""
    # Load data and model
    dataloader = load_data(tensor=True)
    groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear = load_model()
    
    # print("Grounding dino parameters")
    # print(list(groundingdino.parameters()))
    # print("Linear image parameters")
    # print(list(groundingdino_img_linear.parameters()))
    # print("Linear text parameters")
    # print(list(groundingdino_txt_linear.parameters()))

    optimizers = {}
    optimizers["groundingdino"] = torch.optim.Adam(groundingdino.parameters(), lr=hyparams["lr"])
    optimizers["sam"] = torch.optim.Adam(sam.parameters(), lr=hyparams["lr"])
    optimizers["groundingdino_img_linear"] = torch.optim.Adam(groundingdino_img_linear.parameters(), lr=hyparams["lr"])
    optimizers["groundingdino_txt_linear"] = torch.optim.Adam(groundingdino_txt_linear.parameters(), lr=hyparams["lr"])
    optimizers["sam_linear"] = torch.optim.Adam(sam.parameters(), lr=hyparams["lr"])
    
    # Training loop
    groundingdino.train()
    sam.train()
    biomedclip.eval()
    for epoch_num in range(hyparams["epochs"]):
        print("Epoch #{}".format(epoch_num))

        for i, data in enumerate(dataloader):
            print("Batch #{}".format(i))
            # Load data
            images = data["image"]
            image_paths = data["image_path"]
            reports = data["report"]

            for key, optimizer in optimizers.items():
                optimizer.zero_grad()

            # Compute loss
            loss = compute_loss(image_paths, reports, groundingdino, sam, biomedclip, tokenizer, preprocess_train, groundingdino_img_linear, groundingdino_txt_linear, sam_linear)
            loss.backward()
            
            for key, optimizer in optimizers.items():
                optimizer.step()
    
    return groundingdino, sam, groundingdino_img_linear, groundingdino_txt_linear
        
        
def load_data_test():
    dataloader = load_data(tensor=True)

    print("Number of batches:", len(dataloader))
    for i, data in enumerate(dataloader):
        images = data["image"]
        image_paths = data["image_path"]
        report = data["report"]
        # print(image_paths, report)
    print("Passed all tests")


if __name__ == "__main__":
    # load_data_test()

    hyparams = {
        "lr": 1e-4,
        "epochs": 1,
    }
    train(hyparams, None, None)