# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:19:00 2017

@author: Szabolcs
"""

import warnings
import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL

from cntk import load_model, combine

import crop


def create_image(image_file,cropped=True,pad_rate=0.25,save_file='',category='',correct_RGBShift=True):
    img = Image.open(image_file)
    img_square=None
    char_sizes=None
    img=None
    if os.path.exists(image_file):
        try:
            img = Image.open(image_file)
            
            if cropped and img.mode=='RGBA':
                img_square, char_sizes=crop.crop(img,pad_rate=0.25,save_file=save_file,category=category,correct_RGBShift=correct_RGBShift)
            else:
                img_square=img.copy() # 3 channel image
        except:
            print('loading error: '+image_file)    
    if img is not None:
        img.close()
    return img_square, char_sizes

class cnn_classification:
    def __init__(self,model_file=None,im_mean=None, model_output_layer=1):
        # model specific parameters
     
        # 0: Softmax, 1: Unnormalised output layer
        assert model_output_layer in (0,1), "model output layer must be 0 or 1"
        
        self.im_mean=im_mean
        #self.model_name='cnn_model.dnn'
        #model_file=os.path.join(self.param.model_dir,self.model_name)
        print('...loading classification model')
        # ToDo: do checks for image size and num_channel
        
        mod = load_model(model_file)
        nodes=mod.find_all_with_name('')
        
        self.pred  = combine([nodes[model_output_layer]])
        
        self.im_height=mod.arguments[0].shape[1]
        self.im_width=mod.arguments[0].shape[2]
        self.im_channels=mod.arguments[0].shape[0]
    
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
