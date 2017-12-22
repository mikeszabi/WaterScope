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
import numpy as np
import os
from src_train.train_config import train_params
import src_tools.file_helper as fh
#import crop 
from skimage import feature
from skimage.color import rgb2gray

from matplotlib import pyplot as plt


#data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
data_dir=os.path.join(r'E:\OneDrive\WaterScope')


cfg=train_params(data_dir)

save_dir=r'd:\DATA\WaterScope\tmp_binary'

image_list=fh.imagelist_in_depth(cfg.base_imagedb_dir,level=2)
#image_list=fh.imagelist_in_depth(os.path.join(data_dir,'Images','crop_problems'),level=1)

#for image_file in image_list:
#        img = Image.open(image_file)
#        img_square=crop.crop(img,pad_rate=0.25,
#                             save_file=os.path.join(data_dir,'Images','tmp',os.path.basename(image_file)),
#                             category='dummy')


#file_names=[f for f in image_list]
class_names=[os.path.dirname(f).split('\\')[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':image_list,'Class name':class_names})

"""
Filter for classes

"""
#df_filtered=df_db[df_db['Class name']=='_Trash']
#df_filtered=df_db.copy()
#image_file=df_filtered.iloc[1]['Filename']
#image_file=r'e:\OneDrive\WaterScope\Images\crop_problems\0002678_DHM2.0.20151007.000..20151008T113053-0003.png'

#lengths=[]
for i,row in df_db.iterrows():
    
    image_file=row['Filename']
    
    category=row['Class name']
    
    try:    
        save_dir_cat=os.path.join(save_dir,category)
        if not os.path.exists(save_dir_cat):
            os.makedirs(save_dir_cat)
        save_file=os.path.join(save_dir_cat,os.path.basename(image_file))
    
    
        img = Image.open(image_file)
    
        img_rgb=img.convert('RGB')     # if png
         #   img_rgb=img
        
        im = np.asarray(img_rgb,dtype=np.uint8)
        
        gray=rgb2gray(im)
        
    #    pixel_per_micron=crop.get_pixelsize(img)
           
        
    #    print(str(pixel_per_micron)+' '+image_file)
    #    lengths.append(pixel_per_micron)
        
        
        contr=gray.max()-gray.min()
        high_tsh=max(0.1,contr*0.32)
        low_tsh=max(0.005,contr*0.01)
    #    edges1 = feature.canny(gray, sigma=1.5, low_threshold=0.01, high_threshold=0.995, use_quantiles=True)
        edges1 = feature.canny(gray, sigma=2, low_threshold=low_tsh, high_threshold=high_tsh, use_quantiles=False)
    
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))            
        
        ax1.imshow(im)
        ax1.axis('off')  
        
        ax2.imshow(edges1)
        ax2.axis('off')    
     
        
        fig.savefig(save_file)
        plt.close('all')
    #    save_file=os.path.join(save_dir,os.path.basename(image_file))
    #    img_square, char_sizes=crop.crop(img,pad_rate=0.25,save_file=save_file,category='')
    except:
        print('error: '+image_file)

