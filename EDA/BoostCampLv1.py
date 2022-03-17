# %%
from matplotlib import units
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy as np
import pandas as pd

from pathlib import Path
from glob import glob

from functools import reduce

# %%
ls_img = glob('/opt/ml/input/data/train/images/*/*.*')
ls_img = [Path(i) for i in ls_img]
piece = [(i.parent.stem, i.stem, i) for i in ls_img]
piece = [(*i.split("_"), j, k) for i, j, k in piece]

# %%
dic_mask = {'mask':0, 'incorrect':1, 'normal':2}
dic_gender = {'male':0, 'female':1}
dic_age = {'Young':0,'Adult':1,'Old':2}

def cls_age(age):
    dic = {'Young':(0, 30),'Adult':(30, 60),'Old':(60, np.inf)}
    
    for res, ranges in dic.items():
        if ranges[0] <= int(age) < ranges[1]:
            return res
    raise KeyError

def cls_mask(mask):
    dic = {
        'mask':('mask1','mask2','mask3','mask4','mask5'), 
        'incorrect':('incorrect_mask',), 
        'normal':('normal', )
        }
    
    for res, ranges in dic.items():
        if mask in ranges:
            return res
    raise KeyError

def cls_multi(mask, gender, age):
    return 6*dic_mask[mask] + 3*dic_gender[gender] + dic_age[age]
    
# %%
df = pd.DataFrame(piece, columns=['id', 'gender', 'race', 'age_raw', 'mask_raw', 'path'])
df['age'] = df['age_raw'].map(cls_age)
df['mask'] = df['mask_raw'].map(cls_mask)
df['multi'] = [cls_multi(row['mask'], row['gender'], row['age']) for _, row in df.iterrows()]

df_refined = df[['id', 'mask', 'gender', 'age', 'multi', 'path']]

# %%
if __name__ == '__main__':
    UNIT_SIZE = 5
    size = [1,3]
    classes = ['mask', 'gender', 'age']

    size = np.array(size)
    figsize = reduce(lambda x, y: x * y, size) / size * UNIT_SIZE

    fig, ax = plt.subplots(*size, figsize=figsize)
    for idx, item in enumerate(classes):
        sns.countplot(data=df_refined, x=item, ax=ax[idx])

    plt.show()

# %%
def title_func(row):
    return f"id:{row['id']}//multi:{row['multi']}"

def view_img(data, col_img, size=(5,5), UNIT_SIZE=5, title_func=None, save_name=None):
    assert size != (1,1), '무조건 1개 이상의 이미지만 가능.'
    assert len(size)==2, '무조건 2차원 size만 가능.'
    
    size = np.array(size)
    product = reduce(lambda x, y: x * y, size)
    figsize = product / size * UNIT_SIZE

    _, axes = plt.subplots(*size, figsize=figsize, constrained_layout=True)

    it = np.nditer(axes, flags=["refs_ok", 'multi_index'])
    samples = data.sample(n=product)
    for idx, _ in enumerate(it):
        row = samples.iloc[idx]
        ax = axes[it.multi_index]
        
        image = img.imread(row[col_img])
        ax.imshow(image)
        ax.set_axis_off()
        
        if title_func:
            ax.set_title(title_func(row))
    
    if save_name:
        plt.savefig(save_name)
    plt.show()
