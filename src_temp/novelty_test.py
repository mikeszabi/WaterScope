# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:33:56 2018

@author: SzMike
"""

import warnings

import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
import skimage.io as io
import csv
import collections

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
io.use_plugin('pil') # Use only the capability of PIL

from __future__ import print_function
import cntk
cntk.__version__
#from cntk.device import all_devices, gpu
#all_devices()
#gpu(0)

from cntk import load_model, combine
import src_tools.file_helper as fh
import crop

def create_image(image_file,cropped=True,pad_rate=0.25,save_file='',category='',correct_RGBShift=True):
    img = Image.open(image_file)
    if cropped and img.mode=='RGBA':
        img_square, char_sizes=crop.crop(img,pad_rate=0.25,save_file=save_file,category=category,correct_RGBShift=correct_RGBShift)
    else:
        img_square=img.copy() # 3 channel image
    img.close()
    return img_square, char_sizes

class cnn_classification:
    def __init__(self,model_file=None,im_height = 64, im_width  = 64,im_mean=None):
        # model specific parameters
        self.im_height=im_height # ToDo: parameter
        self.im_width=im_width # ToDo: parameter

        self.im_mean=im_mean
        #self.model_name='cnn_model.dnn'
        #model_file=os.path.join(self.param.model_dir,self.model_name)
        print('...loading classification model')
        # ToDo: do checks for image size and num_channel
        
        mod=load_model(model_file)
        nodes=mod.find_all_with_name('')
        self.pred  = combine([nodes[1]])
    
    def classify(self, img, char_sizes=None):
        
        if char_sizes is not None:
            maxl=char_sizes[0].astype('float32')
            minl=char_sizes[1].astype('float32')
   
        if not type(img)==np.ndarray:
            if img.mode=='RGBA':
                img=img.convert('RGB')
            im=np.asarray(img,dtype=np.uint8)
        else:
            im=img.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = img_as_ubyte(resize(im, (self.im_height,self.im_width), order=1))
        if data.ndim==3: 
            rgb_image=data.astype('float32')
            if self.im_mean:
                rgb_image  -= self.im_mean
            bgr_image = rgb_image[..., [2, 1, 0]]
            pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2)) # CHW format
        else:
            gray_image=data.astype('float32')
            pic = np.ascontiguousarray(np.expand_dims(gray_image, axis=0)) # CHW format
       
        if char_sizes is not None:    
            result  = np.round(np.squeeze(self.pred.eval({self.pred.arguments[0]:[pic],
                                             self.pred.arguments[1]:[[maxl,minl]]}))*100)
        else:
            result  = np.round(np.squeeze(self.pred.eval({self.pred.arguments[0]:[pic]}))*100)
        predicted_label=np.argmax(result)
        predicted_prob=max(result)
        
        return predicted_label, predicted_prob
    
    
model_file=r'D:\Projects\WaterScope\model\cnn_model.dnn'
typedict_file=r'D:\Projects\WaterScope\model\type_dict.csv'


type_dict={}
reader =csv.DictReader(open(typedict_file, 'rt'), delimiter=':')
for row in reader:
    type_dict[int(row['label'])]=row['type']
    
ord_type_dict=collections.OrderedDict(sorted(type_dict.items()))

pred=load_model(model_file)

nodes=pred.find_all_with_name('')
output_nodes  = combine([nodes[1]])

cnn_taxon=cnn_classification(model_file,64,64)


image_dir=r'e:\OneDrive\WaterScope\Images\Novelty_RosszulOsztalyozott'
#image_dir=r'e:\OneDrive\WaterScope\Images\Novelty_HelyesenOsztalyozott'

image_list=fh.imagelist_in_depth(image_dir,level=2)

df_db = pd.DataFrame(data={'Filename':image_list})
strength=np.zeros(len(image_list))
label=np.zeros(len(image_list))

for i, image_file in enumerate(image_list):

#    image_file=image_list[1]
    img, char_sizes = create_image(image_file,cropped=True,correct_RGBShift=True)
    label[i], strength[i] = cnn_taxon.classify(img,char_sizes=char_sizes)
    print(strength[i])
    
df_db['label']=label    
df_db['strength']=strength


# save df_db
df_db.to_csv('WrongClass.csv',sep=',')
#df_db.to_csv('CorrectClass.csv',sep=',')


# read
df_wrong=pd.read_csv('WrongClass.csv',sep=',')
df_correct=pd.read_csv('CorrectClass.csv',sep=',')


##

fig = plt.figure()
ax1 = fig.add_subplot(111)

h1=ax1.hist(df_correct['strength'], normed=True, alpha=0.5,label='correct')
h2=ax1.hist(df_wrong['strength'], normed=True, alpha=0.5,label='wrong')

ax1.legend()
plt.show()

tsh=1500
len(df_correct[df_correct['strength']>tsh])/len(df_correct)
len(df_wrong[df_wrong['strength']<tsh])/len(df_wrong)

df_out=df_correct[df_correct['strength']<tsh]

fig = plt.figure()
ax1 = fig.add_subplot(111)

h1=ax1.hist(df_out['label'],bins=range(0,len(ord_type_dict.values())))
ax1.set_xticks(range(0,len(ord_type_dict.values())))
ax1.set_xticklabels(ord_type_dict.values(),rotation=90, rotation_mode="anchor", ha="right")

rate=np.zeros(len(ord_type_dict.values()))
df_out['label'].value_counts()[12]
for i in range(0,len(rate)):
    try:
        rate[i]=df_out['label'].value_counts()[i]/df_correct['label'].value_counts()[i]
        continue
    except:
        print(i)

fig = plt.figure()
ax1 = fig.add_subplot(111)

h1=ax1.bar(range(0,len(rate)),rate)
ax1.set_xticks(range(0,len(ord_type_dict.values())))
ax1.set_xticklabels(ord_type_dict.values(),rotation=90, rotation_mode="anchor", ha="right")

### Examples

df_corr_sort=df_correct.sort_values(by=['strength'])
        
i=2
df_corr_sort.iloc[i]['Filename']
type_dict[df_corr_sort.iloc[i]['label']]

img=mpimg.imread(df_corr_sort.iloc[i]['Filename'])
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.imshow(img)
ax1.set_title(type_dict[df_corr_sort.iloc[i]['label']])

df_wrong_sort=df_wrong.sort_values(by=['strength'])
        
i=-2
df_wrong_sort.iloc[i]['Filename']
type_dict[df_wrong_sort.iloc[i]['label']]

img=mpimg.imread(df_wrong_sort.iloc[i]['Filename'])
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.imshow(img)
ax1.set_title(type_dict[df_wrong_sort.iloc[i]['label']])