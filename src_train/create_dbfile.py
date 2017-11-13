# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:21:20 2017

@author: Szabolcs Mike
Create image db file
Uses class_map
"""


import csv

import pandas as pd
import os
from src_train.train_config import train_params
import src_tools.file_helper as fh


data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
#data_dir=os.path.join(r'E:\OneDrive\WaterScope')
cfg=train_params(data_dir,crop=True)
count_threshold = 200

image_list=fh.imagelist_in_depth(cfg.curdb_dir,level=2)

"""
Class names from folder names
"""
file_names=[f for f in image_list]
class_names=[os.path.dirname(f).split('\\')[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':file_names,'Class name':class_names})

#df_db.to_csv(cfg.db_file,sep=';')

#classes_count=df_db['Class name'].value_counts()
#df_classes = pd.DataFrame(data={'Class name':classes_count.index,'Count':classes_count.values})
#
#df_classes.to_csv(cfg.classnames_file,sep=';')

"""
Group merging - using class_map file
class_map maps the folder names to taxon names
"""
merge_dict={}
with open(cfg.merge_file, mode='r') as infile:
    reader = csv.reader(infile,delimiter=':')
    for rows in reader:
        merge_dict[rows[0]]=rows[1]

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
df_db.replace(to_remove, class_name_low_count, inplace=True)

df_db.to_csv(cfg.db_file,sep=';',index=None)


"""
Do some stats
"""
classes_count=df_db['Class name'].value_counts()
