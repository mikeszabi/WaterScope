# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:54:05 2017

@author: picturio

Test result on test list

"""

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
cfg=train_params(data_dir,crop=True,training_id='20171113')
typedict_file=os.path.join(cfg.train_dir,'type_dict.csv')
model_file=os.path.join(cfg.train_dir,'cnn_model.dnn')


model_file=os.path.join(cfg.train_dir,'cnn_model.dnn')
user='picturio'
imgSize=64
num_classes  = 32

write_misc=False



type_dict={}
reader =csv.DictReader(open(typedict_file, 'rt'), delimiter=':')
for row in reader:
    type_dict[row['label']]=row['type']

sorted_classes= OrderedDict(sorted(type_dict.items(), key=lambda x:x[0]))



# LOAD MODEL
pred=load_model(model_file)


image_mean   = 128


df_test = pd.read_csv(cfg.test_image_list_file,delimiter=';')
samples = {}
contingency_table=np.zeros((num_classes,num_classes))
misclassified=[]
for i, row in df_test.iterrows():
    row['image']
#    i=200
    image_file=row['image']   

    label=row['category']

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
              columns=sorted_classes)  # 1st row as the column names

if write_misc:
    a=[i[1][0] for i in misclassified]
    for misc in misclassified:
        save_file=os.path.join(data_dir,'misc',misc[1][0]+'___'+misc[0])
        # "predicted label"___"original label"
        copyfile(misc[0],save_file)
        print(misc)
    
stats=multiclass_statistics(cont_table,macro=False)

## Calculate statistical measures for multiclass classification
## 1 vs. all single class approach
## in cont_table rows are actual labels, cols are predictions!                
#
#num_classes=cont_table.shape[0]
#n_obs=cont_table.sum().sum()
#
## Allocate memory
#tp=[None]*num_classes # correct inclass lassification
#tn=[None]*num_classes # correct outclass classification - 1 vs. all
#fp=[None]*num_classes # incorrect inclass classification
#fn=[None]*num_classes # incorrect outclass classification
#
## Calculate classification rates        
#tp=[cont_table.iloc[i,i] for i in range(0,num_classes)] # correctly identified
#fp=[-tp[i]+sum(cont_table.iloc[i,:]) for i in range(0,num_classes)] # incorrectly identified class members
#fn=[-tp[i]+sum(cont_table.iloc[:,i]) for i in range(0,num_classes)] # incorrectly identified non-class members
#tn=[n_obs-fp[i]-fn[i]-tp[i] for i in range(0,num_classes)] # correctly identified non-class members
#
## calculate statistics
## recall - same as sensitivity
#
## 1., MACRO - all classes equally weighted
#
## MACRO - all classes equally weighted
#precision=sum([(tp[i])/(tp[i]+fp[i]) for i in range(0,num_classes)])/num_classes
#recall=sum([(tp[i])/(tp[i]+fn[i]) for i in range(0,num_classes)])/num_classes
#specificity=sum([(tn[i])/(tn[i]+fp[i]) for i in range(0,num_classes)])/num_classes
#avg_accuracy=sum([(tp[i]+tn[i])/(tp[i]+fn[i]+fp[i]+tn[i]) for i in range(0,num_classes)])/num_classes
#
## 2., MICRO - larger classes have more weight
#precision=sum([(tp[i]) for i in range(0,num_classes)])/sum([(tp[i]+fp[i]) for i in range(0,num_classes)])
#recall=sum([(tp[i]) for i in range(0,num_classes)])/sum([(tp[i]+fn[i]) for i in range(0,num_classes)])
#specificity=sum([(tn[i]) for i in range(0,num_classes)])/sum([(tn[i]+fp[i]) for i in range(0,num_classes)])
#avg_accuracy=sum([(tp[i]) for i in range(0,num_classes)])/n_obs
#
#beta=1
#fscore=(np.square(beta)+1)*precision*recall/(np.square(beta)*precision+recall)
#
#print