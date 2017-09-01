# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:16:16 2017

@author: SzMike
"""

import warnings

import os
import math 
import crop

import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte

from cntk import load_model

from file_helper import imagelist_in_depth

import skimage.io as io

from skimage import feature
from skimage import morphology
from skimage import measure

io.use_plugin('pil') # Use only the capability of PIL
#%matplotlib qt5
from matplotlib import pyplot as plt

from skimage.color import rgb2gray

import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh


user='SzMike'

data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
db_image_dir=os.path.join(data_dir,'db_images')


image_list_indir = imagelist_in_depth(db_image_dir,level=1)


image_file=image_list_indir[20]

img = Image.open(image_file)

img_rgb=img.convert('RGB')     # if png
#   img_rgb=img

im = np.asarray(img_rgb,dtype=np.uint8)

image_gray = rgb2gray(im)

# LoG
#blobs = blob_log(1-image_gray, max_sigma=30, num_sigma=10, threshold=.05)

# DoG
blobs = blob_dog(1-image_gray, min_sigma=5, threshold=.1)
blobs[:, 2] = blobs[:, 2] * math.sqrt(2)

# DoH
#blobs = blob_doh(image_gray, max_sigma=30, threshold=.01)


fig, axes = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
axes.imshow(im, interpolation='nearest')
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    axes.add_patch(c)
    

areas = [prop.area for prop in props]    
    if areas:       
        prop_large = props[np.argmax(areas)]       
        bb=prop_large.bbox
    else:
        bb=(0,0,gray.shape[0],gray.shape[1])



