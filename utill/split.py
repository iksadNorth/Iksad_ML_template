# %%
import numpy as np
from torch.utils.data import Subset
import sys


this_module = sys.modules[__name__]
# %%
# 이하 내용은 변형 금지.
def _init_case(dataset):
    length = dataset.__len__()
    _case = set(range(length))
    return (_case)

def _make_subset(dataset, train_val_indices):
    return (Subset(dataset, indices) for indices in train_val_indices)

# %%
# 이하 내용이 변형 가능한 공간이다.
def _scrape_random(dataset, num:int) -> list:
    _case = _init_case(dataset)
    stuff = np.random.choice(list(_case), size=num, replace=False)
    _case -= set(stuff)
    return list(stuff), _case

def _split_train_val(dataset, ratio=0.0, method_scrape='_scrape_random'):
    _case = _init_case(dataset)
    method_scrape = getattr(this_module , method_scrape)
    num = int(dataset.__len__() * ratio)
    
    val_indices, _case = method_scrape(dataset, num)
    train_indices = list(_case)
    
    yield _make_subset(dataset, (train_indices, val_indices))

def _split_k_fold(dataset, k, method_scrape='_scrape_random'):
    _case = _init_case(dataset)
    method_scrape = getattr(this_module, method_scrape)
    num = int(dataset.__len__() // k)
    
    tray = []
    while _case.__len__() > num:
        dish, _case = method_scrape(dataset, num)
        tray.append(dish)

    for val_indices in tray:
        train_indices = []
        
        for other in tray:
            if other == val_indices:
                continue
            train_indices += other
        
        yield _make_subset(dataset, (train_indices, val_indices))
