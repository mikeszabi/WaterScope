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
from skimage import filters

io.use_plugin('pil') # Use only the capability of PIL
#%matplotlib qt5
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from skimage.color import rgb2gray
#from skimage.feature import blob_dog

import numpy as np

min_extent_in_micron=20

def crop_edge(gray,pixel_per_micron=1.5):
    contr=gray.max()-gray.min()
    high_tsh=max(0.1,contr*0.32)
    low_tsh=max(0.005,contr*0.01)
#    edges1 = feature.canny(gray, sigma=1.5, low_threshold=0.01, high_threshold=0.995, use_quantiles=True)
    edges1 = feature.canny(gray, sigma=2, low_threshold=low_tsh, high_threshold=high_tsh, use_quantiles=False)

    edges2 = morphology.binary_dilation(edges1,morphology.disk(11))
 
    bb=(0,0,gray.shape[0],gray.shape[1])
    char_sizes=np.asarray((max(gray.shape)/pixel_per_micron,min(gray.shape)/pixel_per_micron))
    
    label_im=measure.label(edges2)
    props = measure.regionprops(label_im,intensity_image=1-gray)
    
    areas = [prop.area for prop in props]   
    max_intensities = [prop.max_intensity  for prop in props] 
    if areas:    
        prop_large = props[np.argmax(max_intensities)]  
        if prop_large.major_axis_length>min_extent_in_micron*pixel_per_micron:
            # min length in micron
            bb=prop_large.bbox
            char_sizes=np.asarray((prop_large.major_axis_length/pixel_per_micron,
                        prop_large.minor_axis_length/pixel_per_micron))

    return bb,char_sizes, edges1

#def crop_blob(gray):
#    # DoG
#    blobs = blob_dog(1-gray, min_sigma=5, threshold=.1)
#    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
#
#    # DoH
#    #blobs = blob_doh(image_gray, max_sigma=30, threshold=.01)
#    
#    if blobs.size>0:       
#        blob_large = blobs[np.argmax(blobs[:, 2]),:]     
#        bb=(blob_large[0]-blob_large[2],blob_large[1]-blob_large[2],2*blob_large[2],2*blob_large[2])
#    else:
#        bb=(0,0,gray.shape[0],gray.shape[1])
#    
#    return bb

def get_pixelsize(img):
    # the linear measure is 20 micron
    line_length=20
    # 4th layer codes the pixelsize
    im = np.asarray(img,dtype=np.uint8)

    im_last=im[:,:,3]

    thresh = filters.threshold_mean(im_last)
    binary = im_last < thresh

    label_img = measure.label(binary)
    regions = measure.regionprops(label_img)

    length=np.NaN
    if len(regions)==1:
        bb = regions[0].bbox
        length=bb[3]-bb[1]
# length in pixel per micron    
        pixel_per_micron=length/line_length
    else:
        pixel_per_micron=1.5
    if pixel_per_micron<1:
        pixel_per_micron=1.5
    return pixel_per_micron
   

def crop(img,pad_rate=0.25,save_file='',category=''):

    pixel_per_micron = get_pixelsize(img)
    
    img_rgb=img.convert('RGB')     # if png
#   img_rgb=img
    
    im = np.asarray(img_rgb,dtype=np.uint8)
    
    gray=rgb2gray(im)
    
    bb, char_sizes, edges1=crop_edge(gray,pixel_per_micron=pixel_per_micron)
#        
    dx=bb[2]-bb[0]
    dy=bb[3]-bb[1]
    rmax=int(max((0.5+pad_rate)*dx,(0.5+pad_rate)*dy))
    o=(int(np.ceil((bb[0]+bb[2])/2)),int(np.ceil((bb[1]+bb[3])/2)))
    bb_square=(max(o[0]-rmax,0),
               max(o[1]-rmax,0),
               min(o[0]+rmax,gray.shape[0]),
               min(o[1]+rmax,gray.shape[1]))
        
    im_cropped = im[bb_square[0]:bb_square[2], bb_square[1]:bb_square[3],:]
        
    if min(im_cropped.shape)>0:
        img_cropped = Image.fromarray(np.uint8(im_cropped))
        img_w, img_h = img_cropped.size
        img_square=Image.new('RGBA', (max(img_cropped.size),max(img_cropped.size)), (255,255,255,255))
        bg_w, bg_h = img_square.size
        offset = (int(np.ceil((bg_w - img_w) / 2)), int(np.ceil((bg_h - img_h) / 2)))
        img_square.paste(img_cropped, offset)
        
        if save_file and category:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
            fig.suptitle(category+'  '+"{:.0f}".format(char_sizes[0])+','+"{:.0f}".format(char_sizes[1]))
        
            ax1.imshow(im)
            #ax1.axis('off')
       
            ax1.add_patch(patches.Rectangle(
                        (bb[1], bb[0]),   # (x,y)
                            bb[3]-bb[1],        # width
                            bb[2]-bb[0],       # height
                            fill=False))
            
            ax2.imshow(np.asarray(img_square))
            #ax2.axis('off')
            
            
            fig.savefig(save_file)
            plt.close('all')
    else:
        img_square=[]
        
    return img_square, char_sizes
        
    