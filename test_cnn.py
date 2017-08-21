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

from cntk import load_model

user='picturio'
imgSize=32
num_classes  = 17

data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
image_dir=os.path.join(data_dir,'cropped_highclass_20170710')

typedict_file=os.path.join(image_dir,'TypeDict_3.csv')
type_dict={}
reader =csv.DictReader(open(typedict_file, 'rt'), delimiter=';')
for row in reader:
    type_dict[row['type']]=row['label']

sorted_classes= [i[0] for i in sorted(type_dict.items(), key=lambda x:x[0].upper())]


user='picturio'
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#output_base_dir=r'd:\DATA\WaterScope'

train_dir=os.path.join(output_base_dir,'Training')

model_file=os.path.join(train_dir,'cnn_model.dnn')
image_dir=os.path.join(output_base_dir,'cropped_highclass_20170710')
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
misclassified=[]
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
    if predicted_label  != label:
        mis_item=[os.path.basename(im_name),
        keysWithValue(type_dict,str(predicted_label)),
        keysWithValue(type_dict,str(label))]
        misclassified.append(mis_item)
#    print(df['wbc'][i])
#    print(result)
#    print(keysWithValue(param.wbc_basic_types,str(mr)))
#    plt.imshow(im)
#    
a=[i[1][0] for i in misclassified]
for misc in misclassified:
    image_file=os.path.join(image_dir,misc[0])    
    save_file=os.path.join(data_dir,'misc',misc[1][0]+'___'+misc[0])
    im=io.imread(image_file)
    io.imsave(save_file,im)
    print(misc)
    
cont_table=pd.DataFrame(data=contingency_table,    # values
              index=sorted_classes,    # 1st column as index
              columns=sorted_classes)  # 1st row as the column names
