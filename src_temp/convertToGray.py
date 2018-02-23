# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:19:49 2017

@author: SzMike
"""

import os
#from PIL import Image
import pandas as pd
from PIL import Image
from src_train.train_config import train_params
import src_tools.file_helper as fh
#import src_tools.file_helper as fh

#==============================================================================
# SET THESE PARAMETERS!
#==============================================================================

training_id='dummy'
curdb_dir='db_cropped_rot'
#data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
data_dir=os.path.join('/','home','mikesz','ownCloud','WaterScope')

# cropped results are saved here
save_dir=os.path.join(data_dir,'Images','db_cropped_blue')

#==============================================================================
# RUN CONFIG
#==============================================================================

cfg=train_params(data_dir,base_db='db_categorized',curdb_dir=curdb_dir,training_id=training_id)



"""
Create image db
"""
image_list=fh.imagelist_in_depth(cfg.curdb_dir,level=1)

class_names=[os.path.dirname(f).split(os.sep)[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':image_list,'Class name':class_names})

#==============================================================================
# COPY to gray
#==============================================================================


char_size_dict={}
for i, image_file in enumerate(image_list):
#   i=1
    image_file=image_list[i]
    
    row=df_db.loc[df_db['Filename'] == image_file]
    
    if not row.empty:
        category=row['Class name'].values[0]
        img = Image.open(image_file)
        img_rgb=img.convert('RGB')
        #img_gray=img.convert('L')
        img_red, img_green, img_blue = img_rgb.split()

        img_blue=Image.merge('RGB',(img_blue,img_blue,img_blue))
        cat_dir=os.path.join(save_dir,category)
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)
        crop_file=os.path.join(cat_dir, os.path.basename(image_file))
        img_blue.save(crop_file)

     
    else:        
        print('not in db: '+image_file)   
