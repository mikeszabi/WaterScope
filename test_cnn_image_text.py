# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:54:05 2017
@author: SzMike
Test result on test list
"""
training_id='20180228'

from shutil import copyfile
import csv
import pandas as pd
import os
import shutil
from collections import OrderedDict
import numpy as np
from PIL import Image
from src_tools.file_helper import check_folder


from src_train.train_config import train_params
from src_train.multiclass_stats import multiclass_statistics
import classifications

curdb_dir='db_cropped_rot'
data_dir=os.path.join('/','home','mikesz','ownCloud','WaterScope')


#==============================================================================
# RUN CONFIG
#==============================================================================

cfg=train_params(data_dir,base_db='db_categorized',curdb_dir=curdb_dir,training_id=training_id)

# model dimensions

write_misc=False
write_folders=True

save_dir=os.path.join(data_dir,'Images','tmp_val')


#==============================================================================
# RUN CONFIG
#==============================================================================

cfg=train_params(data_dir,training_id=training_id)

typedict_file=os.path.join(cfg.train_dir,'type_dict.csv')
model_file=os.path.join(cfg.train_dir,'cnn_model.dnn')




type_dict={}
reader =csv.DictReader(open(typedict_file, 'rt'), delimiter=':')
for row in reader:
    type_dict[row['label']]=row['type']

sorted_classes= OrderedDict(sorted(type_dict.items(), key=lambda x:x[1])).values()



# LOAD MODEL
cnn_class=classifications.cnn_classification(model_file)


num_classes=cnn_class.pred.output.shape[0]

df_test_image = pd.read_csv(cfg.test_image_list_file,delimiter=';')
df_test_text = pd.read_csv(cfg.test_text_list_file,delimiter=';')

samples = {}
contingency_table=np.zeros((num_classes,num_classes))
misclassified=[]

df_images_processed=df_test_image.reset_index(drop=True).copy()
df_images_processed['predicted_type']=None
df_images_processed['prob_taxon']=None
pd.options.mode.chained_assignment = None  # default='warn'

for index, row in df_test_image.iterrows():
#    print(row['image'])
#    i=200
    image_file=row['image']   

    label=row['category']
    maxl=np.asarray([df_test_text.iloc[index]['maxl']]).astype('float32')[0]
    minl=np.asarray([df_test_text.iloc[index]['minl']]).astype('float32')[0]
   
    img=Image.open(image_file)
   
    predicted_label, predicted_prob=cnn_class.classify(img, char_sizes=(maxl, minl))
    contingency_table[label,predicted_label]+=1
    # rows are actual labels, cols are predictions,     

    df_images_processed['predicted_type'][index]=type_dict[str(predicted_label)]
    df_images_processed['prob_taxon'][index]=predicted_prob

    if write_folders:
        cur_dir=os.path.join(save_dir,type_dict[str(predicted_label)])
        check_folder(folder=cur_dir,create=True)
        shutil.copy(image_file, os.path.join(cur_dir,os.path.basename(image_file)))  
 
             
    if predicted_label  != label:
        mis_item=[image_file,
        type_dict[str(predicted_label)],
        type_dict[str(label)]]
        misclassified.append(mis_item)

#    

cont_table=pd.DataFrame(data=contingency_table,    # values
              index=sorted_classes,    # 1st column as index
              columns=sorted_classes) # 1st row as the column names

cont_table.to_csv(os.path.join(cfg.train_dir,'cont_table.csv'))

if write_misc:
    a=[i[1][0] for i in misclassified]
    for misc in misclassified:
        file_name,extension = os.path.splitext(misc[0])
        save_file=os.path.join(data_dir,'Images','misc',misc[1]+'___'+misc[2]+extension)
        # "predicted label"___"original label"
        copyfile(misc[0],save_file)
#        os.remove(save_file)
        print(misc)
    
stats=multiclass_statistics(cont_table,macro=False)

##
"""
# check merged:
print('Class map file: '+os.path.basename(cfg.merge_file))
merge_file_1=os.path.join(r'C:\\Users\\picturio\\OneDrive\\WaterScope\\Images\\cropped_images','class_map_merge.csv')
merge_file_0=os.path.join(r'C:\\Users\\picturio\\OneDrive\\WaterScope\\Images\\cropped_images','class_map.csv')
merge_dict_0={}
with open(merge_file_0, mode='r') as infile:
    reader = csv.reader(infile,delimiter=':')
    next(reader,None) # skip header
    for rows in reader:
        if rows:
            if rows[0][0]!='#':
                merge_dict_0[rows[0]]=rows[1]
                
merge_dict_1={}
with open(merge_file_1, mode='r') as infile:
    reader = csv.reader(infile,delimiter=':')
    next(reader,None) # skip header
    for rows in reader:
        if rows:
            if rows[0][0]!='#':
                merge_dict_1[rows[0]]=rows[1]
                
merge_dict={}
for k,v in merge_dict_0.items():
    merge_dict[v]=merge_dict_1[k]
    
cont_table_man_merge=cont_table.copy()
for k,v in merge_dict.items():
    cont_table_man_merge.rename(columns = {k:v},inplace=True)
    cont_table_man_merge.rename(index = {k:v},inplace=True)
classes_count=df_db['Class name'].value_counts()
cont_table_man_merge.groupby(index)
tmp_1=cont_table_man_merge.groupby(level=0).sum()
tmp_2=pd.DataFrame()
for taxon in tmp_1.index:
    tmp_2[taxon]=tmp_1[[taxon]].sum(axis=1)
# print df.groupby('value')['tempx'].apply(' '.join).reset_index()
"""