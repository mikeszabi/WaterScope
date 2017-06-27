# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:48:12 2017

@author: SzMike
"""

import os
import glob

import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
%matplotlib qt5
from matplotlib import pyplot as plt
from PIL import Image

from skimage import feature
from skimage import morphology
from skimage import measure

from scipy import ndimage

from skimage.color import rgb2gray

import pandas as pd
import numpy as np

user='picturio'
data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#data_dir=r'd:\DATA\WaterScope'

image_dir=os.path.join(data_dir,'original')
crop_dir=os.path.join(data_dir,'cropped')


included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']

db_file=os.path.join(data_dir,'Database.csv')
df = pd.read_csv(db_file,delimiter=';')

image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))


for i, image_file in enumerate(image_list_indir):
#   i=100
    image_file=image_list_indir[i]
    
    label=df.loc[df['Filename'] == os.path.basename(image_file)]
    category=label['Class name'].values[0]
    
    img = Image.open(image_file)
    #im = io.imread(image_file) # read uint8 image
    im = np.asarray(img,dtype=np.uint8)
    gray=rgb2gray(im)
    
    edges1 = feature.canny(gray, sigma=2)
    edges2 = morphology.binary_dilation(edges1,morphology.disk(5))
    
    label_im=measure.label(edges2)
    props = measure.regionprops(label_im)
    
    areas = [prop.area for prop in props]
    
    if areas:
        
        prop_large = props[np.argmax(areas)]
        
        o = np.round(prop_large.centroid ).astype('uint8')
        max_r = np.min((np.min(gray.shape-o),np.min(o)))
        r = np.ceil(np.max(prop_large.bbox[2:4])/2).astype('uint8')
        r = np.min((r,max_r))
        
        im_cropped = im[o[0]-r:o[0]+r, o[1]-r:o[1]+r,:]
    
    else:
        im_cropped=im
        
    img_cropped = Image.fromarray(np.uint8(im_cropped))
    for alpha in [0,90,180,270]:
        rot_cropped=img_cropped.rotate(alpha, expand=True)
        crop_file=os.path.join(crop_dir,category+'__'+str(i)+'__'+str(alpha)+'.png')
        rot_cropped.save(crop_file)
    
    # ToDo: rotate save cropped image
    
    
#    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
#    fig.suptitle(category)
#
#    ax1.imshow(im, cmap=plt.cm.gray)
#    ax1.axis('off')
#    #ax1.set_title('Original', fontsize=20)
#
#    ax2.imshow(edges2, cmap=plt.cm.gray)
#    ax2.axis('off')
#    #ax2.set_title('Binary', fontsize=20)
#    
#    ax3.imshow(roi, cmap=plt.cm.gray)
#    ax3.axis('off')
#    #ax3.set_title('Crop', fontsize=20)
#    
#    fig.savefig(os.path.join(saveDir,os.path.basename(image_file)))
#    plt.close('all')
