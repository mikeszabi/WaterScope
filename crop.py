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

pad_rate=0.25

user='picturio'
data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#data_dir=r'd:\DATA\WaterScope'

image_dir=os.path.join(data_dir,'merged export')
crop_dir=os.path.join(data_dir,'cropped_highclass_20170710')
save_dir=os.path.join(data_dir,'tmp')


included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']

db_file=os.path.join(image_dir,'Database.csv')
df = pd.read_csv(db_file,delimiter=';')

qualities=df['Class quality']


image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
#   i=1
    image_file=image_list_indir[i]
    
    label=df.loc[df['Filename'] == os.path.basename(image_file)]
    
    if not label.empty:
        category=label['Class name'].values[0]
        qual=label['Class quality'].values[0]
        
        if qual=='highclass' or qual=='unclassified':
        
            img = Image.open(image_file)
            #im = io.imread(image_file) # read uint8 image
            im = np.asarray(img,dtype=np.uint8)
#            im_rect=np.zeros((max(im.shape),max(im.shape),im.shape[2]))
#            
#            im_rect
            
            gray=rgb2gray(im)
            
            edges1 = feature.canny(gray, sigma=2)
            edges2 = morphology.binary_dilation(edges1,morphology.disk(5))
            
            label_im=measure.label(edges2)
            props = measure.regionprops(label_im)
            
            areas = [prop.area for prop in props]
            
            if areas:
                
                prop_large = props[np.argmax(areas)]
                
                bb=prop_large.bbox
                
               
            else:
                bb=(0,0,gray.shape[0],gray.shape[1])
                
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
                
                for alpha in [0,90,180,270]:
                    img_rot=img_square.rotate(alpha, expand=True)
                    crop_file=os.path.join(crop_dir,category+'__'+str(i)+'__'+str(alpha)+'.png')
                    img_rot.save(crop_file)
                            
                
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
                fig.suptitle(category)
            
                ax1.imshow(im, cmap=plt.cm.gray)
                ax1.axis('off')
                #ax1.set_title('Original', fontsize=20)
            
                ax2.imshow(edges2, cmap=plt.cm.gray)
                ax2.axis('off')
                #ax2.set_title('Binary', fontsize=20)
                
                ax3.imshow(im_cropped, cmap=plt.cm.gray)
                ax3.axis('off')
                #ax3.set_title('Crop', fontsize=20)
                
                fig.savefig(os.path.join(save_dir,os.path.basename(image_file)))
                plt.close('all')

