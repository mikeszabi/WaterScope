# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:54:05 2017

@author: picturio
"""


import warnings
import pandas as pd
import os
import numpy as np
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
from skimage.transform import resize
from skimage import img_as_ubyte

from cntk import load_model


imgSize=32
num_classes  = 28



user='picturio'
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#output_base_dir=r'd:\DATA\WaterScope'

train_dir=os.path.join(output_base_dir,'Training')

model_file=os.path.join(train_dir,'cnn_model.dnn')
image_dir=os.path.join(output_base_dir,'cropped')
train_dir=os.path.join(output_base_dir,'Training')
image_list_file=os.path.join(train_dir,'images_test.csv')
model_file=os.path.join(train_dir,'cnn_model.dnn')


# LOAD MODEL
pred=load_model(model_file)


image_mean   = 128


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

df = pd.read_csv(image_list_file,delimiter=';')
samples = {}
contingency_table=np.zeros((num_classes,num_classes))
for i, im_name in enumerate(df['image']):
#    i=200
    image_file=os.path.join(image_dir,im_name)    
#    image_file=r'C:\Users\SzMike\OneDrive\WBC\DATA\Training\Train\ne_50.png'

#    wbc_type='0'
#    for bt in param.wbc_basic_types:
#        if bt in df['category'][i]:
    label=df['category'][i]

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
    contingency_table[predicted_label,label]+=1
#    print(df['wbc'][i])
#    print(result)
#    print(keysWithValue(param.wbc_basic_types,str(mr)))
#    plt.imshow(im)
#    

