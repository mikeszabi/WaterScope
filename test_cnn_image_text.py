# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:54:05 2017

@author: SzMike

Test result on test list

"""
#training_id='20171120-All'
#num_classes  = 23

from shutil import copyfile
import warnings
import csv
import pandas as pd
import os
from collections import OrderedDict
import numpy as np
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
from skimage.transform import resize
from skimage import img_as_ubyte
from cntk import load_model

from src_train.train_config import train_params
from src_train.multiclass_stats import multiclass_statistics


data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
cfg=train_params(data_dir,crop=True,training_id=training_id)
typedict_file=os.path.join(cfg.train_dir,'type_dict.csv')
model_file=os.path.join(cfg.train_dir,'cnn_model.dnn')


model_file=os.path.join(cfg.train_dir,'cnn_model.dnn')
user='picturio'
imgSize=64

write_misc=False



type_dict={}
reader =csv.DictReader(open(typedict_file, 'rt'), delimiter=':')
for row in reader:
    type_dict[row['label']]=row['type']

sorted_classes= OrderedDict(sorted(type_dict.items(), key=lambda x:x[1])).values()



# LOAD MODEL
pred=load_model(model_file)


image_mean   = 128


df_test_image = pd.read_csv(cfg.test_image_list_file,delimiter=';')
df_test_text = pd.read_csv(cfg.test_text_list_file,delimiter=';')

samples = {}
contingency_table=np.zeros((num_classes,num_classes))
misclassified=[]
for i, row in df_test_image.iterrows():
#    print(row['image'])
#    i=200
    image_file=row['image']   

    label=row['category']
    maxl=np.asarray([df_test_text.iloc[i]['maxl']]).astype('float32')[0]
    minl=np.asarray([df_test_text.iloc[i]['minl']]).astype('float32')[0]
   

    im=io.imread(image_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = img_as_ubyte(resize(im, (imgSize,imgSize), order=1))
    rgb_image=data.astype('float32')
    rgb_image  -= image_mean
    bgr_image = rgb_image[..., [2, 1, 0]]
    pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

       
    result  = np.round(np.squeeze(pred.eval({pred.arguments[0]:[pic],
                                             pred.arguments[1]:[[maxl,minl]]}))*100)
    predicted_label=np.argmax(result)
    contingency_table[label,predicted_label]+=1
    # rows are actual labels, cols are predictions,                  
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
merge_dict={}
with open(cfg.merge_file, mode='r') as infile:
    reader = csv.reader(infile,delimiter=':')
    next(reader,None) # skip header
    for rows in reader:
        if rows:
            if rows[0][0]!='#':
                merge_dict[rows[0]]=rows[1]

for k,v in merge_dict.items():
    cont_table.replace(k,v,inplace=True)
classes_count=df_db['Class name'].value_counts()

# print df.groupby('value')['tempx'].apply(' '.join).reset_index()

"""