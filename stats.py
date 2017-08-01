# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:02:07 2017

@author: picturio
"""

import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from PIL import Image
import os

import numpy as np


user='picturio'
data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
image_dir=os.path.join(data_dir,'merged export')
#data_dir=r'd:\DATA\WaterScope'


db_file=os.path.join(image_dir,'Database.csv')
typedict_file=os.path.join(image_dir,'TypeDict.csv')

df = pd.read_csv(db_file,delimiter=';')


df_high=df[df['Class quality'] == 'highclass']
classes=df_high['Class name'].value_counts()


fig = plt.figure(1, figsize=(72., 40.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 9),  # creates 2x2 grid of axes
                 axes_pad=1,  # pad between axes in inch.
                 )
samp=[]
for i, cl in enumerate(classes.keys()):
    df_high_cl=df[df['Class name'] == cl]
    samp.append(df_high_cl.sample(n=1))
    
    img = Image.open(os.path.join(image_dir,samp[i].iloc[0]['Filename']))
    img.thumbnail((100,100))
    im = np.asarray(img,dtype=np.uint8)
    grid[i].imshow(im, cmap=plt.cm.gray)
    grid[i].axis('off')
    grid[i].set_title(samp[i].iloc[0]['Class name'], fontsize=8)
    grid[i].text(100,100,str(classes[cl]))
