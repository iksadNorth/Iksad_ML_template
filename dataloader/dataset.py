# %%
import sys

import numpy as np
from torch.utils.data import Dataset, Subset

from torchvision import datasets
from torchvision.transforms import ToTensor

from EDA.BoostCampLv1 import df_refined, dic_mask, dic_gender, dic_age
from PIL import Image

# %%
class BaseDataset(Dataset):
    def __init__(self):
        self.classes = None
        super(BaseDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError 

    def __len__(self):
        raise NotImplementedError
    
# %%
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

# %%
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

# %%
class BoostCampLv1(Dataset):
    def __init__(self, transform=None, label:str='multi'):
        assert label in ['mask', 'gender', 'age', 'multi'], 'Label 오류. KeyError'
        
        current_module = sys.modules[__name__]
        super(BoostCampLv1, self).__init__()
        if label == 'multi':
            self.classes = list(range(18))
        else:
            self.classes = list(getattr(current_module, f'dic_{label}').keys())
        self.dataset = df_refined[['path', label]]

    def __getitem__(self, index):
        path, label = tuple(self.dataset.iloc[index].tolist())
        return Image.open(path), label

    def __len__(self):
        return len(self.dataset)


# %%
