# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 22:44:15 2017

@author: SzMike
"""


import os
import glob


from PIL import Image

import pandas as pd
import numpy as np

#import crop

from skimage import feature
from skimage import morphology
from skimage import measure
from skimage import filters

import skimage.io as io
import matplotlib.patches as patches

from skimage.feature import blob_dog, blob_log, blob_doh


io.use_plugin('pil') # Use only the capability of PIL
#%matplotlib qt5
from matplotlib import pyplot as plt

from skimage.color import rgb2gray
from scipy import ndimage as ndi

import math

#from cfg import *

import cv2
def mask_color_img(img, mask, color=[0, 255, 255], alpha=0.3):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1]. 

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask] = color
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return(out)


included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
user='SzMike'
#user='picturio'
data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#data_dir=r'd:\DATA\WaterScope'


db_image_dir=os.path.join(data_dir,'db_images')

db_file=os.path.join(db_image_dir,'Database.csv')

save_dir=os.path.join(data_dir,'edge')
save_dir2=os.path.join(data_dir,'blob')


cur_image_dir=db_image_dir
cur_image_dir=os.path.join(data_dir,'crop_problems')

#included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']

df = pd.read_csv(db_file,delimiter=';')

image_list_indir = []

for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(cur_image_dir, ext)))


for i, image_file_full in enumerate(image_list_indir):
#   i=1
    image_file_full=image_list_indir[i]
    
#    image_file_full=os.path.join(db_image_dir,
#                             '0000279_DHM2.0.20151007.000..20151012T104022-0001.png')
    
    label=df.loc[df['Filename'] == os.path.basename(image_file_full)]
    if not label.empty:
        category=label['Class name'].values[0]
    else:
        category='none'
    
    img = Image.open(image_file_full)
    im = np.asarray(img,dtype=np.uint8)
    
    img_rgb=img.convert('RGB')     # if png
    #   img_rgb=img
        
    im = np.asarray(img_rgb,dtype=np.uint8)
    
    gray=im[:,:,0] #rgb2gray(im) #im[:,:,0] #rgb2gray(im)
    #plt.imshow(gray, cmap='gray')
   
#    kernel = np.real(filters.gabor_kernel(frequency=0.2, theta=0,sigma_x=2, sigma_y=2))
#    filtered = ndi.convolve(gray, kernel, mode='wrap')
    #plt.imshow(filtered)
    
    #mask=gray<filters.threshold_otsu(gray)*1.25
    #mask=morphology.binary_dilation(mask,morphology.disk(3))
    
    edges1 = feature.canny(gray, sigma=1.5, low_threshold=0.01, high_threshold=0.995, use_quantiles=True)
    #plt.imshow(edges2)
    
    #edges1 = feature.canny(gray, sigma=2)
    #edges2 = morphology.binary_closing(edges1,morphology.disk(11))
    edges2 = morphology.binary_dilation(edges1,morphology.disk(11))
    
    save_file=os.path.join(save_dir,os.path.basename(image_file_full))
    
    qq=mask_color_img(im, edges2, color=[255, 255, 0], alpha=0.5)
    #qq=mask_color_img(im, mask, color=[255, 0, 0], alpha=0.8)
    
    im_save = Image.fromarray(qq.astype('uint8'))
    im_save.save(save_file)
    
    label_im=measure.label(edges2)
    props = measure.regionprops(label_im)
         
    bb=(0,0,gray.shape[0],gray.shape[1])
   
    areas = [prop.area for prop in props]    
    l = [prop.major_axis_length  for prop in props]  
    if areas:    
        if max(l)>max(gray.shape[0],gray.shape[1])*0.25:
            prop_large = props[np.argmax(areas)]       
            bb=prop_large.bbox
    
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    fig.suptitle(category)

    ax1.imshow(qq, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.add_patch(patches.Rectangle(
                (bb[1], bb[0]),   # (x,y)
                    bb[3]-bb[1],        # width
                    bb[2]-bb[0],       # height
                    fill=False))         
    
    ax1.axis('off')
             
    fig.savefig(save_file)
    plt.close('all')