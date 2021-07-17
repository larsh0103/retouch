from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms
import random
import torch


def get_transforms():
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=(512, 512)),
            transforms.Resize(128),
            transforms.ToTensor()
        ]
    )


def get_target_transforms():
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=(512, 512)),
            transforms.ToTensor()
        ]
    )

class RetouchDataset(Dataset):
    def __init__(self,original_dir, retouch_dir, image_start, image_stop, transform = None , target_transform = None):
        self.original_dir = original_dir
        self.retouch_dir = retouch_dir
        self.image_start = image_start 
        self.image_stop = image_stop
        self.indexes = list(np.arange(image_start,image_stop))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.image_stop - self.image_start

    
    def __getitem__(self,idx):
        file_name = "{0:05d}.png".format(self.indexes[idx])
        image_dir = "{0:05d}".format((self.indexes[idx]//1000)*1000) 
        sample = Image.open(os.path.join(self.original_dir,image_dir,file_name))
        gt = sample
        ## Uncomment for retouching not just SR
        # gt = Image.open(os.path.join(self.retouch_dir,image_dir,file_name))

        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            sample = self.transform(sample)
            random.seed(seed)
            torch.manual_seed(seed)
            gt = self.target_transform(gt)
        
        return sample,gt

    
        




