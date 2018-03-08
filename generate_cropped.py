# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:48:12 2017

@author: SzMike
"""
import os
import time
from pathlib import Path
#from PIL import Image
import pandas as pd
import classifications
from src_train.train_config import train_params
import src_tools.file_helper as fh
#import src_tools.file_helper as fh

#==============================================================================
# SET THESE PARAMETERS!
#==============================================================================

training_id='20180308'
curdb_dir='db_cropped_rot'
#data_dir=os.path.join('C:','Users','picturio','OneDrive','WaterScope')
#data_dir=os.path.join('E:','OneDrive','WaterScope')
data_dir=os.path.join('/','home','mikesz','ownCloud','WaterScope')
# cropped results are saved here
#save_dir=os.path.join('D:','DATA','WaterScope','tmp_problem')
save_dir=os.path.join('/','home','mikesz','Data','WaterScope','tmp_cropped')
#==============================================================================
# RUN CONFIG
#==============================================================================

cfg=train_params(data_dir,base_db='db_categorized',curdb_dir=curdb_dir,training_id=training_id)


#==============================================================================
# Create database using folder names
#==============================================================================
image_list=fh.imagelist_in_depth(cfg.base_imagedb_dir,level=1)

"""
Class names from folder names
"""
class_names=[os.path.dirname(f).split(os.sep)[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':image_list,'Class name':class_names})
#df_db=df_db[df_db['Class name']=='Others']

#==============================================================================
# Read database to dataframe
#==============================================================================
#df = pd.read_csv(cfg.db_file,delimiter=';')
#image_list=list(df_db['Filename'])

char_size_dict={}
for i, image_file in enumerate(image_list):
#   i=1
    print(i)
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
            t=time.time()
            img_square, char_sizes=classifications.create_image(image_file,cropped=True,
                                                    save_file=save_file,
                                                    category=category,
                                                    correct_RGBShift=False)
            print('time elapsed: '+str(time.time()-t))

        except:
            print('loading error: '+image_file)
        
        if img_square is not None and char_sizes is not None:
            char_size_dict[image_file]=char_sizes
            cat_dir=os.path.join(cfg.curdb_dir,category)
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
                        
            for alpha in [0,90,180,270]:
                img_rot=img_square.rotate(alpha, expand=True)
                crop_file=os.path.join(cfg.curdb_dir,category,
                                       Path(image_file).resolve().stem+'__'+str(alpha)+'.png')
                img_rot.save(crop_file)
                img_rot.close()
#            crop_file=os.path.join(cfg.curdb_dir,category,os.path.basename(image_file))
#            img_square.save(crop_file)    

            img_square.close()
                
        else:
            print('crop problem: '+image_file)   
    else:        
        print('not in db: '+image_file)   

"""
Saving characteristic sizes
"""
class_names=[]
file_names=[]
maxl=[]
minl=[]

for k,v in char_size_dict.items():
    class_names.append(os.path.dirname(k).split(os.sep)[-1]) 
    file_names.append(os.path.basename(k))
    maxl.append(v[0])
    minl.append(v[1])
    
pd_char_size=pd.DataFrame(data={'Filename':file_names,'Class name':class_names,'maxl':maxl,'minl':minl})
                      
size_file=os.path.join(cfg.base_imagedb_dir,'char_sizes.txt')
pd_char_size.to_csv(size_file,sep=';',index=None)
