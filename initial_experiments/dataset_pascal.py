import torchvision.datasets.voc as voc
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
import pickle
import h5py
import numpy as np
from utils import explore_tensor
import pdb

class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """
    def __init__(
        self, 
        root, 
        year='2012', 
        image_set='train', 
        download=False, 
        transform=None, 
        target_transform=None,
        h5_file='/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/initial_experiments/data/pascal_train.h5',
    ):
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform
        )
        self.h5_file = h5_file
    

    def __getitem__(self, i):
        anno = self.parse_voc_xml(ET_parse(self.annotations[i]).getroot())["annotation"]
        image_path = self.images[i]
        
        labels = {}
        # for key in ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse", "motorbike", "person", "plant", "sheep", "sofa", "train", "monitor"]:
        #     labels[key] = []

        for obj in anno["object"]:
            label = obj["name"]
            if label == "tvmonitor":
                label = "monitor"
            elif label == "diningtable":
                label = "table"
            elif label == "pottedplant":
                label = "plant"
            if label not in labels:
                labels[label] = []
            
            labels[label].append([
                int(obj["bndbox"]["xmin"]), 
                int(obj["bndbox"]["ymin"]), 
                int(obj["bndbox"]["xmax"]), 
                int(obj["bndbox"]["ymax"])
            ])
        
        # Get preprocessed image and report from h5 file
        with h5py.File(self.h5_file,'r') as h5f:
            # Get processed images
            img_dset_gd = h5f['img_gd']
            img_gd = img_dset_gd[i]
            img_dset_biovil = h5f['img_biovil']
            img_biovil = img_dset_biovil[i]

        # Get orignal image size
        original_img_size = [int(anno["size"]["height"]), int(anno["size"]["width"])]
        original_img_size = torch.tensor(original_img_size)

        data = {
            "image_path": image_path,
            "labels": labels,
            "image_gd": img_gd,
            "image_biovil": img_biovil,
            "original_img_size": original_img_size
        }

        return data
    
    def __len__(self):
        return len(self.annotations)


def collate_fn(batch):
    """Customized collate function to handle varying sized labels in pascal."""
    batched_data = {}
    # Since different samples have varying sized labels, we store them as a list
    batched_data['labels'] = [item['labels'] for item in batch]

    # For other keys, use the default collate function (i.e., store as tensor)
    for key in batch[0].keys():
        if key != 'labels':
            batched_data[key] = default_collate([item[key] for item in batch])

    return batched_data


def load_data(batch_size=16, h5_file='/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/initial_experiments/data/pascal_train.h5', num_workers=0):
    """Get dataloader for training.
    """
    dataset = PascalVOC_Dataset(
        root="/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/",
        image_set="train",
        h5_file=h5_file,
        transform=None
    )
    # collate function helps to stack samples with different length into a single batch
    # may lead to slowness
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)


class UnitTest:
    def __init__(self):
        pass

    def view_dataset(self):
        dataset = PascalVOC_Dataset(
            root="/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/",
            image_set="train",
            h5_file='/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/initial_experiments/data/pascal_train.h5',
            transform=None
        )
        print(dataset[0])
        pdb.set_trace()


    def load_data_test(self):
        dataloader = load_data(batch_size=16)

        print("Number of batches:", len(dataloader))
        for i, data in enumerate(tqdm(dataloader)):
            # print(data)
            pdb.set_trace()
        print("Passed all tests")
    

if __name__ == "__main__":
    unittest = UnitTest()
    # unittest.view_dataset()
    unittest.load_data_test()