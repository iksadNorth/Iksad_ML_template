# %%
from config.adapter import adapter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torchvision import models as repo_pretrained_model
# %%

class PretrainedModel(nn.Module):
    
    # 이 모델의 위한 환경변수들.
    NUM_CLASSES_ASSUMED=1000
    HIDDEN_LAYER_FEATURES=100
    def __init__(self, num_classes, model_using:str='resnet18', pretrained:bool=True , freeze:bool=True,**kwargs):
        ERROR_MASSAGE = f"해당 model_using({model_using})은 torchvision.model에 존재하지 않음."
        assert hasattr(repo_pretrained_model, model_using), ERROR_MASSAGE
        
        super().__init__()
        # 파라미터들 필드로 선언.
        self.num_classes = num_classes
        self.model_using = model_using
        self.pretrained = pretrained
        
        # 출처
        # https://github.com/pytorch/vision/tree/main/torchvision/models
        # 해당 모델들의 num_classes가 항상 1000이라는 가정하에 설계됨.
        # 위 가정은 꼼꼼히 확인한 것이 아니기 때문에 오류가 날 수 있음.
        self.model = getattr(repo_pretrained_model, model_using)
        # print(f"{model_using} :: {self.model.__qualname__}")
        
        
        self.net_pretrained = self.model(pretrained, **kwargs)
        # print(f"net_pretrained :: {self.net_pretrained}")
        
        
        if freeze:
            self._freeze()
        
        # 해당 모델들의 num_classes가 항상 1000이라는 가정하에 설계됨. 
        # ONLY (num_classes==1000)
        self.net_mlp = nn.Sequential(
            nn.Linear(self.NUM_CLASSES_ASSUMED, self.HIDDEN_LAYER_FEATURES, bias=True),
            nn.GELU(),
            nn.Linear(self.HIDDEN_LAYER_FEATURES, self.num_classes, bias=True),
        )
        
        # pretrained module과 MLP의 결합.
        self.sequential = nn.Sequential(
            self.net_pretrained,
            self.net_mlp,
        )
        # print(f"final_model :: {self.sequential}")


    def forward(self, x):
        x = self.sequential(x)
        return x
    
    
    def _freeze(self):
        net = self.net_pretrained
        
        for params in net.parameters():
            params.require_grad = False
    
    
    def _melt(self):
        net = self.net_pretrained
        
        for params in net.parameters():
            params.require_grad = True
    
    
    def _melt_gradually(self, idx_elapsed:int):
        net = self.net_pretrained
        
        # 고안 예정.
        pass