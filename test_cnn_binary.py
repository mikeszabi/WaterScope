# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:54:05 2017

@author: picturio
"""


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

from cntk import load_model

user='picturio'
imgSize=32
num_classes  = 2

write_misc=False

data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
db_image_dir=os.path.join(data_dir,'db_images')


user='picturio'
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#output_base_dir=r'd:\DATA\WaterScope'

train_dir=os.path.join(output_base_dir,'Training')

model_file=os.path.join(train_dir,'cnn_model_binary.dnn')

train_dir=os.path.join(output_base_dir,'Training')
image_list_file=os.path.join(train_dir,'images_test_binary.csv')

df_db = pd.read_csv(db_file,delimiter=';')

type_dict={'Trash':'0','Object':'1'}

# LOAD MODEL
pred=load_model(model_file)


image_mean   = 128


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

# RUN TEST
df_test = pd.read_csv(image_list_file,delimiter=';')
contingency_table=np.zeros((num_classes,num_classes))
misclassified=[]

df_res = pd.DataFrame(columns=['Filename','orig_quality','orig_category','label','predicted_label'])


for i, im_name in enumerate(df_test['image']):
#    i=200
    image_file=os.path.join(db_image_dir,im_name)    
#    image_file=r'C:\Users\SzMike\OneDrive\WBC\DATA\Training\Train\ne_50.png'

    label=df_test['category'][i]

    im=io.imread(image_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = img_as_ubyte(resize(im, (imgSize,imgSize), order=1))
    rgb_image=data.astype('float32')
    rgb_image  -= image_mean
    bgr_image = rgb_image[..., [2, 1, 0]]
    pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
    
#    rgb_image = np.asarray(Image.open(image_file), dtype=np.float32) - 128
#    bgr_image = rgb_image[..., [2, 1, 0]]
#    pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
       
    result  = np.round(np.squeeze(pred.eval({pred.arguments[0]:[pic]}))*100)
    predicted_label=np.argmax(result)
    contingency_table[label,predicted_label]+=1
    row=df_db.loc[df_db['Filename'] == os.path.basename(image_file)]
    # rows are actual labels, cols are predictions,                  
    if predicted_label  != label:
        mis_item=[os.path.basename(im_name),
        keysWithValue(type_dict,str(predicted_label)),
        row['Class name'].values[0]]
        misclassified.append(mis_item)

    row=df_db.loc[df_db['Filename'] == os.path.basename(image_file)]
    df_temp=pd.DataFrame({'Filename':row['Filename'].values[0],
                        'orig_quality':row['Class quality'].values[0],
                        'orig_category':row['Class name'].values[0],
                        'label':label,
                        'predicted_label':predicted_label},index=[i])
    df_res=pd.concat([df_res,df_temp])      
#    

cont_table=pd.DataFrame(data=contingency_table,    # values
              index=type_dict.keys(),    # 1st column as index
              columns=type_dict.keys())  # 1st row as the column names

if write_misc:
    for misc in misclassified:
        image_file=os.path.join(db_image_dir,misc[0])    
        save_file=os.path.join(data_dir,'misc',misc[1][0]+'___'+misc[2]+'__'+misc[0])
        # "predicted label"___"original label"
        im=io.imread(image_file)
        io.imsave(save_file,im)
        print(misc)
    


# Calculate statistical measures 1vsAll
num_classes=cont_table.shape[0]
n_obs=cont_table.sum().sum()


tp=[None]*num_classes # correct classification
tn=[None]*num_classes # 
# True positive for i-th class; 1 vs. all
        
tp=[cont_table.iloc[i,i] for i in range(0,num_classes)] # correctly identified
fp=[-tp[i]+sum(cont_table.iloc[i,:]) for i in range(0,num_classes)] # incorrectly identified class members
fn=[-tp[i]+sum(cont_table.iloc[:,i]) for i in range(0,num_classes)] # incorrectly identified non-class members
tn=[n_obs-fp[i]-fn[i]-tp[i] for i in range(0,num_classes)] # correctly identified non-class members

# MACRO
beta=1

avg_accuracy=sum([(tp[i]+tn[i])/(tp[i]+fn[i]+fp[i]+tn[i]) for i in range(0,num_classes)])/num_classes
# MACRO - all classes equally weighted
precision=sum([(tp[i])/(tp[i]+fp[i]) for i in range(0,num_classes)])/num_classes
recall=sum([(tp[i])/(tp[i]+fn[i]) for i in range(0,num_classes)])/num_classes
# MICRO - larger classes have more weight
precision=sum([(tp[i]) for i in range(0,num_classes)])/sum([(tp[i]+fp[i]) for i in range(0,num_classes)])
recall=sum([(tp[i]) for i in range(0,num_classes)])/sum([(tp[i]+fn[i]) for i in range(0,num_classes)])

fscore=(np.square(beta)+1)*precision*recall/(np.square(beta)*precision+recall)

##

dd=df_res.groupby('orig_quality')
q_qual=dd.agg([np.mean,len])
#q_qual=dd.describe().loc[['count','mean']]
print(q_qual)
dd=df_res.groupby('orig_category')
q_cat=dd.agg([np.mean,len])
print(q_cat)

