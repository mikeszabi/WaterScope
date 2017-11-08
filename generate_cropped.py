# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:48:12 2017

@author: SzMike
"""

import os
import glob
import csv
#from PIL import Image
import pandas as pd
import classifications
import src_train.train_config as cfg
import src_tools.file_helper as fh
#import src_tools.file_helper as fh


# cropped results are saved here
save_dir=os.path.join(cfg.data_dir,'Images','tmp')
# crop-original map
crop_map_file=os.path.join(cfg.data_dir,'crop_map.csv')

#==============================================================================
# Read database to dataframe
#==============================================================================
df = pd.read_csv(cfg.db_file,delimiter=';')

image_list=fh.imagelist_in_depth(cfg.imagedb_dir,level=2)


crop_map={}

for i, image_file in enumerate(image_list):
#   i=1
    image_file=image_list[i]
    
    label=df.loc[df['Filename'] == image_file]
    
    if not label.empty:
        category=label['Class name'].values[0]
        #qual=label['Class quality'].values[0]
        
        try:
            img_square=classifications.create_image(image_file,cropped=True,save_file=os.path.join(save_dir,os.path.basename(image_file)),
                             category=category)
        except:
            print('loading error: '+image_file)
        
        if img_square:
                        
            for alpha in [0,90,180,270]:
                img_rot=img_square.rotate(alpha, expand=True)
                crop_file=os.path.join(cfg.imagecrop_dir,category+'__'+str(i)+'__'+str(alpha)+'.png')
                img_rot.save(crop_file)
                crop_map[os.path.basename(crop_file)]=os.path.basename(image_file)
                        
                
                

out = open(crop_map_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['cropped','original'])
w.writeheader()
for key, value in crop_map.items():
    w.writerow({'cropped' : key, 'original' : value})
out.close()