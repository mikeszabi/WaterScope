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
    return img_square

class cnn_classification:
    def __init__(self,model_file=None,im_size=64,im_mean=128):
        # model specific parameters
        self.im_size=im_size # ToDo: parameter
        self.im_mean=im_mean
        #self.model_name='cnn_model.dnn'
        #model_file=os.path.join(self.param.model_dir,self.model_name)
        print('...loading classification model')
        self.pred=load_model(model_file)
    
    def classify(self, img):
        
        if not type(img)==np.ndarray:
            im=np.asarray(img.convert('RGB'),dtype=np.uint8)
        else:
            im=img.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = img_as_ubyte(resize(im, (self.im_size,self.im_size), order=1))
        rgb_image=data.astype('float32')
        rgb_image  -= self.im_mean
        bgr_image = rgb_image[..., [2, 1, 0]]
        pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
    
       
        result  = np.round(np.squeeze(self.pred.eval({self.pred.arguments[0]:[pic]}))*100)
        predicted_label=np.argmax(result)
        predicted_prob=max(result)
        
        return predicted_label, predicted_prob
