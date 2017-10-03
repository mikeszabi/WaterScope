# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:54:05 2017

@author: picturio
"""

import glob

import warnings
import csv
import pandas as pd
import os

import numpy as np
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
from skimage.transform import resize
from skimage import img_as_ubyte
from cfg import *

#from cntk import load_model

#import classifications

user='picturio'
imgSize=32
num_classes  = 2

write_misc=False

data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
db_image_dir=os.path.join(data_dir,'db_images')
db_file=os.path.join(db_image_dir,'Database.csv')
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#output_base_dir=r'd:\DATA\WaterScope'
#train_dir=os.path.join(output_base_dir,'Training')
#image_list_file=os.path.join(train_dir,'images_test_binary.csv')

#model_file=os.path.join(train_dir,'cnn_model_binary.dnn')
model_file=os.path.join(r'.\model','cnn_model_binary.dnn')

df_db = pd.read_csv(db_file,delimiter=';')

type_dict={'Trash':'0','Object':'1'}


# LOAD MODEL
pred=load_model(model_file)
image_mean   = 128
#cnn=classifications.cnn_classification(model_file)





def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

# RUN TEST
# filter df
#df_filtered=df_db[(df_db['Class quality']!='highclass')]

df_filtered=df_db

image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(db_image_dir, ext)))

df_res = pd.DataFrame(columns=['Filename','orig_quality','orig_category','predicted_label'])


for i, image_file in enumerate(image_list_indir):
#   i=1
    image_file=image_list_indir[i]

    
    row=df_filtered.loc[df_filtered['Filename'] == os.path.basename(image_file)]

    if not row.empty:
#        im = classifications.create_image(image_file,cropped=False)
#        predicted_label, prob = cnn.classify(im)
        im=io.imread(image_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = img_as_ubyte(resize(im, (imgSize,imgSize), order=1))
        rgb_image=data.astype('float32')
        rgb_image  -= image_mean
        bgr_image = rgb_image[..., [2, 1, 0]]
        pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
       
        result  = np.round(np.squeeze(pred.eval({pred.arguments[0]:[pic]}))*100)
        predicted_label=np.argmax(result)
        class_name=row['Class name'].values[0]
        row=df_db.loc[df_db['Filename'] == os.path.basename(image_file)]
        df_temp=pd.DataFrame({'Filename':row['Filename'].values[0],
                        'orig_quality':row['Class quality'].values[0],
                        'orig_category':row['Class name'].values[0],
                        'predicted_label':predicted_label},index=[i])
        
        df_res=pd.concat([df_res,df_temp])          




##

dd=df_res.groupby('orig_quality')
q_qual=dd.agg([np.mean,len])
#q_qual=dd.describe().loc[['count','mean']]
print(q_qual)
dd=df_res.groupby('orig_category')
q_cat=dd.agg([np.mean,len])
print(q_cat)

