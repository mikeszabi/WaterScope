# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:48:12 2017

@author: SzMike
"""

import os
from pathlib import Path
#from PIL import Image
import pandas as pd
import classifications
from src_train.train_config import train_params
import src_tools.file_helper as fh
#import src_tools.file_helper as fh

data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
#data_dir=os.path.join(r'E:\OneDrive\WaterScope')


cfg=train_params(data_dir)

# cropped results are saved here
save_dir=os.path.join(data_dir,'Images','tmp')

#==============================================================================
# Create database using folder names
#==============================================================================
image_list=fh.imagelist_in_depth(cfg.imagedb_dir,level=1)

"""
Class names from folder names
"""
class_names=[os.path.dirname(f).split('\\')[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':image_list,'Class name':class_names})

#==============================================================================
# Read database to dataframe
#==============================================================================
#df = pd.read_csv(cfg.db_file,delimiter=';')
#image_list=list(df_db['Filename'])
image_list=fh.imagelist_in_depth(cfg.imagedb_dir,level=1)


for i, image_file in enumerate(image_list):
#   i=1
    image_file=image_list[i]
    
    row=df_db.loc[df_db['Filename'] == image_file]
    
    if not row.empty:
        category=row['Class name'].values[0]
        #qual=label['Class quality'].values[0]
        
        try:
            save_dir_cat=os.path.join(save_dir,category)
            if not os.path.exists(save_dir_cat):
                os.makedirs(save_dir_cat)
            save_file=os.path.join(save_dir_cat,os.path.basename(image_file))
            img_square=classifications.create_image(image_file,cropped=True,
                                                    save_file=save_file,
                                                    category=category)
        except:
            print('loading error: '+image_file)
        
        if img_square:
            cat_dir=os.path.join(cfg.imagecrop_dir,category)
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
                        
            for alpha in [0,90,180,270]:
                img_rot=img_square.rotate(alpha, expand=True)
                crop_file=os.path.join(cfg.imagecrop_dir,category,
                                       Path(image_file).resolve().stem+'__'+str(alpha)+'.png')
                img_rot.save(crop_file)
        else:
            print('crop problem: '+image_file)   
    else:        
        print('not in db: '+image_file)   
                        