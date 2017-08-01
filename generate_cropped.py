# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:48:12 2017

@author: SzMike
"""

import os
import glob
import csv


from PIL import Image

import pandas as pd
import numpy as np

import crop


user='picturio'
data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#data_dir=r'd:\DATA\WaterScope'

image_dir=os.path.join(data_dir,'merged export')
crop_dir=os.path.join(data_dir,'cropped_highclass_20170710')
save_dir=os.path.join(data_dir,'tmp')

crop_map_file=os.path.join(image_dir,'crop_map.csv')

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']

db_file=os.path.join(image_dir,'Database.csv')
df = pd.read_csv(db_file,delimiter=';')

qualities=df['Class quality']


image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

crop_map={}

for i, image_file in enumerate(image_list_indir):
#   i=1
    image_file=image_list_indir[i]
    
    label=df.loc[df['Filename'] == os.path.basename(image_file)]
    
    if not label.empty:
        category=label['Class name'].values[0]
        qual=label['Class quality'].values[0]
        
        if qual=='highclass' or qual=='unclassified':
        
            img = Image.open(image_file)
            im = np.asarray(img,dtype=np.uint8)
            img_square=crop.crop(img,
                                 save_file=os.path.join(save_dir,os.path.basename(image_file)),
                                 category=category)
            
            if img_square:
                            
                for alpha in [0,90,180,270]:
                    img_rot=img_square.rotate(alpha, expand=True)
                    crop_file=os.path.join(crop_dir,category+'__'+str(i)+'__'+str(alpha)+'.png')
                    img_rot.save(crop_file)
                    crop_map[os.path.basename(crop_file)]=os.path.basename(image_file)
                            
                
                

out = open(crop_map_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['cropped','original'])
w.writeheader()
for key, value in crop_map.items():
    w.writerow({'cropped' : key, 'original' : value})
out.close()