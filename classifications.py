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

from cntk import load_model

import crop


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

def create_image(image_file):
    img = Image.open(image_file)
    img_square=crop.crop(img)
    im=np.asarray(img_square)
    return im

class cnn_classification:
    def __init__(self,model_file=None):
        # model specific parameters
        self.img_size=32 # ToDo: parameter
        self.img_mean=128
        #self.model_name='cnn_model.dnn'
        #model_file=os.path.join(self.param.model_dir,self.model_name)
        print('...loading classification model')
        self.pred=load_model(model_file)
    
    def classify(self, im):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = img_as_ubyte(resize(im, (self.img_size,self.img_size), order=1))
        rgb_image=data.astype('float32')
        rgb_image  -= self.img_mean
        bgr_image = rgb_image[..., [2, 1, 0]]
        pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
    
       
        result  = np.round(np.squeeze(self.pred.eval({self.pred.arguments[0]:[pic]}))*100)
        predicted_label=np.argmax(result)
        #predicted_type=keysWithValue(type_dict_2,str(predicted_label))
        predicted_prob=max(result)
        
        return predicted_label, predicted_prob
