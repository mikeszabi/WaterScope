# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:19:00 2017

@author: Szabolcs
"""

import warnings

import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL

from cntk import load_model

import crop


def create_image(image_file,cropped=True,pad_rate=0.25,save_file='',category=''):
    img = Image.open(image_file)
    if cropped:
        img_square, char_sizes=crop.crop(img,pad_rate=0.25,save_file=save_file,category=category)
    else:
        img_square=img.copy()
    return img_square, char_sizes

class cnn_classification:
    def __init__(self,model_file=None,image_height = 64, image_width  = 64,im_mean=None):
        # model specific parameters
        self.image_height=image_height # ToDo: parameter
        self.image_width=image_width # ToDo: parameter

        self.im_mean=im_mean
        #self.model_name='cnn_model.dnn'
        #model_file=os.path.join(self.param.model_dir,self.model_name)
        print('...loading classification model')
        self.pred=load_model(model_file)
    
    def classify(self, img, char_sizes=None):
        
        if char_sizes:
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
            data = img_as_ubyte(resize(im, (self.image_height,self.image_width), order=1))
        if data.ndim==3: 
            rgb_image=data.astype('float32')
            if self.im_mean:
                rgb_image  -= self.im_mean
            bgr_image = rgb_image[..., [2, 1, 0]]
            pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2)) # CHW format
        else:
            gray_image=data.astype('float32')
            pic = np.ascontiguousarray(np.expand_dims(gray_image, axis=0)) # CHW format
       
        if char_sizes:    
            result  = np.round(np.squeeze(self.pred.eval({self.pred.arguments[0]:[pic],
                                             self.pred.arguments[1]:[[maxl,minl]]}))*100)
        else:
            result  = np.round(np.squeeze(self.pred.eval({self.pred.arguments[0]:[pic]}))*100)
        predicted_label=np.argmax(result)
        predicted_prob=max(result)
        
        return predicted_label, predicted_prob
