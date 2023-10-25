import torchvision.datasets.voc as voc
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
import pickle
import pdb
import numpy as np

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
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
        
        # self.preprocess()

        with open(self.root + "pascalvoc_dataset.pkl", "rb") as f:
            self.my_data = pickle.load(f)
    

    def preprocess(self):
        """Preprocess the dataset s.t. each sample if (image_path, label, list of all bbox for that label).
        """
        self.my_data = []

        # Loop through all images
        for i in tqdm(range(len(self.images))):
            anno = self.parse_voc_xml(ET_parse(self.annotations[i]).getroot())["annotation"]
            image_path = self.images[i]
            
            labels = {}
            
            for obj in anno["object"]:
                label = obj["name"]
                if label not in labels:
                    labels[label] = []
                labels[label].append([
                    int(obj["bndbox"]["xmin"]), 
                    int(obj["bndbox"]["ymin"]), 
                    int(obj["bndbox"]["xmax"]), 
                    int(obj["bndbox"]["ymax"])
                ])
            for label in labels:
                data = {
                    "image_path": image_path,
                    "label": label,
                    "bbox": labels[label],
                }
                
                self.my_data.append(data)
        
        # Pad bboxs to have same shape
        bbox_shapes = [len(data["bbox"]) for data in self.my_data]
        max_dim0_shape = max(bbox_shapes)

        for i in range(len(self.my_data)):
            data = self.my_data[i]
            bboxs = data["bbox"]
            for j in range(max_dim0_shape - len(bboxs)):
                bboxs.append([-1.,-1.,-1.,-1.])
            data["bbox"] = torch.Tensor(bboxs)
        
        with open(self.root + "pascalvoc_dataset.pkl", "wb") as f:
            pickle.dump(self.my_data, f)
    
    def __getitem__(self, index):
        data = self.my_data[index]
        data["bbox"] = torch.Tensor(data["bbox"])
        return data
    
    def __len__(self):
        return len(self.my_data)


def load_data(batch_size=16, num_workers=0):
    """Get dataloader for training.
    """
    dataset = PascalVOC_Dataset(
        root="/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/",
        image_set="train",
        transform=None
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


class UnitTest:
    def __init__(self):
        pass

    def view_dataset(self):
        dataset = PascalVOC_Dataset(
            root="/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/",
            image_set="train",
            transform=None
        )
        print(dataset[0])


    def load_data_test(self):
        dataloader = load_data(batch_size=16, num_workers=2)

        print("Number of batches:", len(dataloader))
        for i, data in enumerate(tqdm(dataloader)):
            pass
        print("Passed all tests")
    

if __name__ == "__main__":
    unittest = UnitTest()
    unittest.view_dataset()
    unittest.load_data_test()