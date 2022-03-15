import numpy as np
from torch.utils.data import Dataset, Subset

from torchvision import datasets
from torchvision.transforms import ToTensor

# 위 import 구문 꼭 그대로 쓰셔야 합니다!!!

class BaseDataset(Dataset):
    def __init__(self):
        self.classes = None
        super(BaseDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError 

    def __len__(self):
        raise NotImplementedError
    

class CIFAR100(Dataset):
    def __init__(self, root:str, train:bool=True, transform=None, target_transform=None, download:bool=False):
        self.dataset = datasets.CIFAR100(root, train, transform, target_transform, download)
        super(CIFAR100, self).__init__()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
    def __getattr__(self, attribute_name):
        return  self.dataset.__getattribute__(attribute_name)

class CIFAR10(Dataset):
    def __init__(self, root:str, train:bool=True, transform=None, target_transform=None, download:bool=False):
        self.dataset = datasets.CIFAR10(root, train, transform, target_transform, download)
        super(CIFAR10, self).__init__()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
    def __getattr__(self, attribute_name):
        return  self.dataset.__getattribute__(attribute_name)
    