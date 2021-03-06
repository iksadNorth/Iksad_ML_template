# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from config.adapter import adapter


# %%
def save(path, net, param=False):    
    if param:
        torch.save(net.state_dict(), path)
    else:
        torch.save(net, path)
    
def load(path, net=None):    
    if net:
        net.load_state_dict(torch.load(path))
        return None
    else:
        return torch.load(path)



class SaveTool():
    @adapter
    def __init__(self, path, net, save_frequency=1, param=False) -> None:
        self.best_score = -np.inf
        
        self.param = param
        self.period = save_frequency
        self.period = int(self.period)
        self.path = path
        self.net = net
    
    def save_per_epoch(self, epoch):
        # 주기적으로 모델 저장.
        # "{self.path}/epoch_{epoch}.pth"
        if epoch % self.period == 0:
            save(f"{self.path}/epoch_{epoch}.pth", self.net)

    def save_best(self, score):
        # 최고 성능 갱신 시, 모델 저장.
        # "{self.path}/best.pth"
        if self.best_score < score:
            save(f"{self.path}/best.pth", self.net)
            self.best_score = score