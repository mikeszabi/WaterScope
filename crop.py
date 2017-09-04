# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 20:36:47 2017

@author: picturio
"""



from PIL import Image

import skimage.io as io

from skimage import feature
from skimage import morphology
from skimage import measure

io.use_plugin('pil') # Use only the capability of PIL
#%matplotlib qt5
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from skimage.color import rgb2gray
from skimage.feature import blob_dog

import numpy as np



def crop_edge(gray):
    edges1 = feature.canny(gray, sigma=1.5, low_threshold=0.01, high_threshold=0.995, use_quantiles=True)

    edges2 = morphology.binary_dilation(edges1,morphology.disk(11))
    
    label_im=measure.label(edges2)
    props = measure.regionprops(label_im)
    
    areas = [prop.area for prop in props]    
    if areas:       
        prop_large = props[np.argmax(areas)]       
        bb=prop_large.bbox
    else:
        bb=(0,0,gray.shape[0],gray.shape[1])
    return bb

def crop_blob(gray):
    # DoG
    blobs = blob_dog(1-gray, min_sigma=5, threshold=.1)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    # DoH
    #blobs = blob_doh(image_gray, max_sigma=30, threshold=.01)
    
    if blobs.size>0:       
        blob_large = blobs[np.argmax(blobs[:, 2]),:]     
        bb=(blob_large[0]-blob_large[2],blob_large[1]-blob_large[2],2*blob_large[2],2*blob_large[2])
    else:
        bb=(0,0,gray.shape[0],gray.shape[1])
    
    return bb

    

def crop(img,pad_rate=0.25,save_file='',category=''):

    img_rgb=img.convert('RGB')     # if png
#   img_rgb=img
    
    im = np.asarray(img_rgb,dtype=np.uint8)
    
    gray=rgb2gray(im)
    
    bb=crop_blob(gray)
    
#    edges1 = feature.canny(gray, sigma=1.5, low_threshold=0.15, high_threshold=0.25)
#    #edges1 = feature.canny(gray, sigma=2)
#    edges2 = morphology.binary_dilation(edges1,morphology.disk(5))
#    
#    label_im=measure.label(edges2)
#    props = measure.regionprops(label_im)
#    
#    areas = [prop.area for prop in props]    
#    if areas:       
#        prop_large = props[np.argmax(areas)]       
#        bb=prop_large.bbox
#    else:
#        bb=(0,0,gray.shape[0],gray.shape[1])
#        
    dx=bb[2]-bb[0]
    dy=bb[3]-bb[1]
    dmax=int(max((0.5+pad_rate)*dx,(0.5+pad_rate)*dy))
    o=(int(np.ceil((bb[0]+bb[2])/2)),int(np.ceil((bb[1]+bb[3])/2)))
    bb_square=(max(o[0]-dmax,0),
               max(o[1]-dmax,0),
               min(o[0]+dmax,gray.shape[0]),
               min(o[1]+dmax,gray.shape[1]))
        
    im_cropped = im[bb_square[0]:bb_square[2], bb_square[1]:bb_square[3],:]
        
    if min(im_cropped.shape)>0:
        img_cropped = Image.fromarray(np.uint8(im_cropped))
        img_w, img_h = img_cropped.size
        img_square=Image.new('RGBA', (max(img_cropped.size),max(img_cropped.size)), (0,0,0,0))
        bg_w, bg_h = img_square.size
        offset = (int(np.ceil((bg_w - img_w) / 2)), int(np.ceil((bg_h - img_h) / 2)))
        img_square.paste(img_cropped, offset)
        
        if save_file and category:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
            fig.suptitle(category)
        
            ax1.imshow(im, cmap=plt.cm.gray)
            ax1.axis('off')
            ax1.add_patch(patches.Rectangle(
                        (bb[0], bb[1]),   # (x,y)
                            bb[2],        # width
                            bb[3],       # height
                            fill=False))          
            #ax1.set_title('Original', fontsize=20)
        
#            ax2.imshow(edges2, cmap=plt.cm.gray)
#            ax2.axis('off')
            #ax2.set_title('Binary', fontsize=20)
            
            ax2.imshow(im_cropped, cmap=plt.cm.gray)
            ax2.axis('off')
            
            
            fig.savefig(save_file)
            plt.close('all')
    else:
        img_square=[]
        
    return img_square
        
    