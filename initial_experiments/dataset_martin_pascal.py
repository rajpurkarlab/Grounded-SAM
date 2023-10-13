import torchvision.datasets.voc as voc
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)
        
    
    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)

    
def transform(image):
    """
    Default transform function
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # imagenet mean and std
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    return transform(image)


def load_data(batch_size=16, num_workers=0):
    """Get dataloader for training.
    """
    dataset = PascalVOC_Dataset(
        root="/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/",
        image_set="train",
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


class UnitTest:
    def __init__(self):
        pass

    def load_data_test(self):
        dataloader = load_data(batch_size=4)

        print("Number of batches:", len(dataloader))
        for i, data in enumerate(tqdm(dataloader)):
            print(type(data))
        print("Passed all tests")
    

if __name__ == "__main__":
    # unittest = UnitTest()
    # unittest.load_data_test()

    dataset = PascalVOC_Dataset(
        root="/n/data1/hms/dbmi/rajpurkar/lab/Grounded-SAM/datasets/pascal/",
        image_set="train",
        transform=transform
    )
    print(len(dataset))
    print(dataset[10])