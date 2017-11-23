# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:21:20 2017

@author: Szabolcs Mike
Create image db file
Uses class_map
"""

training_id='20171121-All'

import csv
import numpy as np
import pandas as pd
import os
from src_train.train_config import train_params
import src_tools.file_helper as fh


data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
#data_dir=os.path.join(r'E:\OneDrive\WaterScope')
cfg=train_params(data_dir,crop=True,training_id=training_id)
count_threshold = 75*4

image_list=fh.imagelist_in_depth(cfg.curdb_dir,level=2)

"""
Class names from folder names
"""
file_names=[f for f in image_list]
class_names=[os.path.dirname(f).split('\\')[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':file_names,'Class name':class_names})


classes_count=df_db['Class name'].value_counts()

#df_db.to_csv(cfg.db_file,sep=';')

#classes_count=df_db['Class name'].value_counts()
#df_classes = pd.DataFrame(data={'Class name':classes_count.index,'Count':classes_count.values})
#
#df_classes.to_csv(cfg.classnames_file,sep=';')

"""
Adding size info
"""
size_file=os.path.join(cfg.imagedb_dir,'char_sizes.txt')

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
merge_dict={}
with open(cfg.merge_file, mode='r') as infile:
    reader = csv.reader(infile,delimiter=':')
    next(reader,None) # skip header
    for rows in reader:
        if rows:
            if rows[0][0]!='#':
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

classes_count=df_db['Class name'].value_counts()

print('classes removed:')
print(to_remove)

"""
Do some stats
"""
classes_count=df_db['Class name'].value_counts()
