# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:54:05 2017

@author: picturio
"""


import warnings

import os
import csv
#import tools
import crop

import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte

from cntk import load_model

user='picturio'
imgSize=32
num_classes  = 16



#typedict_2_file=os.path.join(image_dir,'TypeDict_2.csv')


user='picturio'

data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
orig_image_dir=os.path.join(data_dir,'cropped_highclass_20170710')

output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')

out_file=os.path.join(image_dir,'class_res.csv')

train_dir=os.path.join(output_base_dir,'Training')

model_file=os.path.join(train_dir,'cnn_model.dnn')

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']


typedict_3_file=os.path.join(orig_image_dir,'TypeDict_3.csv')
type_dict_3={}
reader =csv.DictReader(open(typedict_3_file, 'rt'), delimiter=';')
for row in reader:
    type_dict_3[row['type']]=row['label']

# LOAD MODEL
pred=load_model(model_file)


image_mean   = 128


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)


image_list_indir = tools.imagelist_in_depth(image_dir,level=2)

preds=[]

for i, im_name in enumerate(image_list_indir):
#    i=0
#  im_name=image_list_indir[i]
    image_file=os.path.join(image_dir,im_name)    

    img = Image.open(image_file)
    img_square=crop.crop(img)
    im=np.asarray(img_square)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = img_as_ubyte(resize(im, (imgSize,imgSize), order=1))
    rgb_image=data.astype('float32')
    rgb_image  -= image_mean
    bgr_image = rgb_image[..., [2, 1, 0]]
    pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
    
       
    result  = np.round(np.squeeze(pred.eval({pred.arguments[0]:[pic]}))*100)
    predicted_label=np.argmax(result)
    predicted_type=keysWithValue(type_dict_2,str(predicted_label))
    predicted_prob=max(result)
    
    one_pred=(image_file,predicted_type,predicted_prob)
    preds.append(one_pred)

#  
out = open(out_file, 'wt')
w = csv.writer(out, delimiter=';')
w.writerow(['file','type','prob'])
for one_pred in preds:
    w.writerow([one_pred[0],one_pred[1][0],one_pred[2]])
out.close()  
