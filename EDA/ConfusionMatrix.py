from random import sample
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image

import numpy as np
import pandas as pd
from einops import rearrange
from functools import reduce


class Analsis():
    def __init__(self, preds, answers, data, classes, path_artifact=None, period=1) -> None:
        self.answers = answers
        self.preds = preds
        self.data = data
        
        self.path_artifact = path_artifact
        self.period = period
        self.period = int(self.period)
        
        
        self.df = pd.DataFrame({'answers': self.answers.cpu().numpy().tolist(), 'preds': self.preds.cpu().numpy().tolist(), 'data': self.data.cpu().numpy().tolist()})
        self.df_incorrect = self.df[self.df['answers'] != self.df['preds']]
        
        self.classes = classes
        self.dict_classes = {i : v for i, v in enumerate(self.classes)}

        self.answers = np.array(self.answers)
        self.preds = np.array(self.preds)
        
    def confusion_matrix(self, epoch, fontsize=14, annotsize=14):
        if epoch % self.period == (self.period-1):
            fig, ax = plt.subplots(1,1, figsize=(18,15))

            cf_matrix = confusion_matrix(self.answers, self.preds, labels=list(self.dict_classes.keys()))
            sns.heatmap(cf_matrix, annot=True, annot_kws={"size":annotsize})

            # confusion_matrix의 첫 파라미터는 y축으로 두 번째 파라미터는 x축으로 감.
            ax.set_xlabel('pred', fontsize=fontsize)
            ax.set_ylabel('answer', fontsize=fontsize)
            
            ticks = list(self.dict_classes.values())
            ax.set_xticklabels(ticks)
            ax.set_yticklabels(ticks)
            
            # save confusion_matrix
            if self.path_artifact:
                plt.savefig(f"{self.path_artifact}/confusion_matrix_epoch_{epoch}.png")
                
            plt.show()
        
    def label_incorrected(self, epoch, size=(5,5), UNIT_SIZE=5): 
        if epoch % self.period == (self.period-1):       
            assert size != (1,1), '무조건 1개 이상의 이미지만 가능.'
            assert len(size)==2, '무조건 2차원 size만 가능.'
            
            size = np.array(size)
            product = reduce(lambda x, y: x * y, size)
            figsize = product / size * UNIT_SIZE

            _, axes = plt.subplots(*size, figsize=figsize, constrained_layout=True)
            
            it = np.nditer(axes, flags=["refs_ok", 'multi_index'])
            samples = self.df_incorrect.sample(product)
            for idx, _ in enumerate(it):
                ax = axes[it.multi_index]
                row = samples.iloc[idx]
                
                img = np.array(row['data'])
                img = rearrange(img, 'c h w -> h w c')
                ax.imshow(img)
                
                pred , answer = self.dict_classes[row['preds']], self.dict_classes[row['answers']]
                
                ax.set_title(f"pred: {pred}\nanswer: {answer}")                
                ax.set_axis_off()
            
            if self.path_artifact:
                plt.savefig(f"{self.path_artifact}/label_incorrected_epoch_{epoch}.png")
            
            plt.show()
        