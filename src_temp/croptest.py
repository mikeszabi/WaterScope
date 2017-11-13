# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:38:50 2017

@author: SzMike
"""

import imp
import sys
#imp.reload(sys.modules['crop'])

from PIL import Image
import pandas as pd
import os
from src_train.train_config import train_params
import src_tools.file_helper as fh
import crop 


#data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
data_dir=os.path.join(r'E:\OneDrive\WaterScope')


cfg=train_params(data_dir)

save_dir=os.path.join(r'E:\OneDrive\WaterScope','Images','tmp')

image_list=fh.imagelist_in_depth(cfg.imagedb_dir,level=2)
image_list=fh.imagelist_in_depth(os.path.join(data_dir,'Images','crop_problems'),level=1)

for image_file in image_list:
        img = Image.open(image_file)
        img_square=crop.crop(img,pad_rate=0.25,
                             save_file=os.path.join(data_dir,'Images','tmp',os.path.basename(image_file)),
                             category='dummy')


#file_names=[f for f in image_list]
class_names=[os.path.dirname(f).split('\\')[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':image_list,'Class name':class_names})

"""
Filter for classes

"""
#df_filtered=df_db[df_db['Class name']=='_Trash']
df_filtered=df_db.copy()
image_file=df_filtered.iloc[1]['Filename']
image_file=r'e:\OneDrive\WaterScope\Images\crop_problems\0002678_DHM2.0.20151007.000..20151008T113053-0003.png'

lengths=[]
for i,d in df_db.iterrows():
    
    image_file=d['Filename']


    img = Image.open(image_file)
    
    pixel_per_micron=crop.get_pixelsize(img)
       
    
    print(str(pixel_per_micron)+' '+image_file)
    lengths.append(pixel_per_micron)
    

    save_file=os.path.join(save_dir,os.path.basename(image_file))
    img_square, char_sizes=crop.crop(img,pad_rate=0.25,save_file=save_file,category='')

    
