# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:21:20 2017

@author: Szabolcs Mike
Create image db file
"""
import csv

import pandas as pd
import os
import src_train.train_config as cfg
import src_tools.file_helper as fh

image_list=fh.imagelist_in_depth(cfg.imagedb_dir,level=2)

file_names=[f for f in image_list]
class_names=[os.path.dirname(f).split('\\')[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':file_names,'Class name':class_names})

#df_db.to_csv(cfg.db_file,sep=';')

classes_count=df_db['Class name'].value_counts()
df_classes = pd.DataFrame(data={'Class name':classes_count.index,'Count':classes_count.values})

df_classes.to_csv(cfg.classnames_file,sep=';')

"""
Group merging
"""
merge_dict={}
with open(cfg.merge_file, mode='r') as infile:
    reader = csv.reader(infile,delimiter=';')
    for rows in reader:
        merge_dict[rows[1]]=rows[2]

for k,v in merge_dict.items():
    df_db.replace(k,v,inplace=True)
classes_count=df_db['Class name'].value_counts()

"""
Eliminating groups with low counts - adding them to a rest container
"""
class_name_low_count='_Others'
elim_dict={}

count_threshold = 50 # Anything that occurs less than this will be removed.
value_counts = df_db['Class name'].value_counts() # Entire DataFrame 
to_remove = value_counts[value_counts <= count_threshold].index
df_db.replace(to_remove, class_name_low_count, inplace=True)
classes_count=df_db['Class name'].value_counts()

df_db.to_csv(cfg.db_file,sep=';')


"""
Do some stats
"""
classes=df_db['Class name'].value_counts()
