# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:38:50 2017

@author: SzMike
"""

import imp
import sys
# imp.reload(sys.modules['crop'])

from PIL import Image
import pandas as pd
import numpy as np
import os
import time

from src_train.train_config import train_params
import src_tools.file_helper as fh

from matplotlib import pyplot as plt
import matplotlib.patches as patches


import crop


#data_dir=os.path.join('C:','Users','picturio','OneDrive','WaterScope')
data_dir=os.path.join('E:','OneDrive','WaterScope')
#data_dir=os.path.join('/','home','mikesz','ownCloud','WaterScope')


cfg=train_params(data_dir,base_db='db_categorized',curdb_dir='crop_problems')


save_dir=os.path.join('D:\\','DATA','WaterScope','tmp_problem')
#save_dir=os.path.join('/','home','mikesz','Data','WaterScope','tmp_problem')

#image_list=fh.imagelist_in_depth(cfg.base_imagedb_dir,level=2)
image_list=fh.imagelist_in_depth(cfg.curdb_dir,level=2)

#for image_file in image_list:
#        img = Image.open(image_file)
#        img_square=crop.crop(img,pad_rate=0.25,
#                             save_file=os.path.join(data_dir,'Images','tmp',os.path.basename(image_file)),
#                             category='dummy')


#file_names=[f for f in image_list]
class_names=[os.path.dirname(f).split(os.sep)[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':image_list,'Class name':class_names})

"""
Filter for classes

"""
#df_filtered=df_db[df_db['Class name']=='_Trash']
#df_filtered=df_db.copy()
#image_file=df_filtered.iloc[1]['Filename']
#image_file=r'e:\OneDrive\WaterScope\Images\crop_problems\0002678_DHM2.0.20151007.000..20151008T113053-0003.png'

#lengths=[]
char_size_dict={}
for i, image_file in enumerate(image_list):
#   i=5
    image_file=image_list[i]
    
    row=df_db.loc[df_db['Filename'] == image_file]
    
    if not row.empty:
        category=row['Class name'].values[0]
    
        try:    
            save_dir_cat=os.path.join(save_dir,category)
            if not os.path.exists(save_dir_cat):
                os.makedirs(save_dir_cat)
            save_file=os.path.join(save_dir_cat,os.path.basename(image_file))
        
        
            img = Image.open(image_file)
#            print(img.mode)
            print(i)
            if img.mode!='RGBA':
                img.close()
                os.remove(image_file)
                continue
                
            t=time.time()
            img_square, char_sizes = crop.crop(img,pad_rate=0.25,save_file=save_file,category='',correct_RGBShift=True)
            print('time elapsed: '+str(time.time()-t))

            
            img.close()
            img_square.close()

           
        except:
            print('error: '+image_file)

