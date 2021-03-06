# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:21:20 2017

@author: Szabolcs Mike
Create image db file
Uses class_map
"""

import csv
import numpy as np
import pandas as pd
import os
from src_train.train_config import train_params
import src_tools.file_helper as fh

#==============================================================================
# SET THESE PARAMETERS!
#==============================================================================

count_threshold = 75*4

training_id='20180308'
curdb_dir='db_cropped_rot'
data_dir=os.path.join('/','home','mikesz','ownCloud','WaterScope')


#==============================================================================
# RUN CONFIG
#==============================================================================

cfg=train_params(data_dir,base_db='db_categorized',curdb_dir=curdb_dir,training_id=training_id)


"""
Class names from folder names
"""
image_list=fh.imagelist_in_depth(cfg.curdb_dir,level=2)

file_names=[f for f in image_list]
class_names=[os.path.dirname(f).split(os.sep)[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':file_names,'Class name':class_names})

df_db=df_db[df_db['Class name']!='Others'] 
df_db=df_db[df_db['Class name']!='_Artefact']

classes_count=df_db['Class name'].value_counts()

#df_db.to_csv(cfg.db_file,sep=';')

#classes_count=df_db['Class name'].value_counts()
#df_classes = pd.DataFrame(data={'Class name':classes_count.index,'Count':classes_count.values})
#
#df_classes.to_csv(cfg.classnames_file,sep=';')

"""
Adding size info
"""
size_file=os.path.join(cfg.base_imagedb_dir,'char_sizes.txt')

df_char_size=pd.read_csv(size_file,delimiter=';')
minl=[]
maxl=[]
for i, row in df_db.iterrows():
    file_name,ext=os.path.splitext(os.path.basename(row['Filename']))
    file_name=file_name.split('__')[0]+ext
    class_id=row['Class name']
    matching_row=df_char_size[(df_char_size['Class name']==class_id) & (df_char_size['Filename']==file_name)]
    minl.append(int(np.round(matching_row['minl'].values[0])))
    maxl.append(int(np.round(matching_row['maxl'].values[0])))
df_db['minl']=minl
df_db['maxl']=maxl


"""
Group merging - using class_map file
class_map maps the folder names to taxon names
"""
print('Class map file: '+os.path.basename(cfg.merge_file))
merge_dict=fh.read_merge_dict(cfg.merge_file)

for k,v in merge_dict.items():
    df_db.replace(k,v,inplace=True)
classes_count=df_db['Class name'].value_counts()

"""
Eliminating groups with low counts - adding them to a rest container (like _Others)
"""
class_name_low_count='Others.Others.Others'
elim_dict={}

value_counts = df_db['Class name'].value_counts() # Entire DataFrame 
to_remove = value_counts[value_counts <= count_threshold].index
if not to_remove.empty:
    df_db.replace(to_remove, class_name_low_count, inplace=True)

df_db.to_csv(cfg.db_file,sep=';',index=None)

classes_count=df_db['Class name'].value_counts()

print('classes removed:')
print(to_remove)

"""
Do some stats
"""
classes_count=df_db['Class name'].value_counts()
