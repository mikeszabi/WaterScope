# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:38:50 2017

@author: SzMike
"""

import imp
import sys
# imp.reload(sys.modules['crop'])

from PIL import Image
import pandas as pd
import numpy as np
import os
from src_train.train_config import train_params
import src_tools.file_helper as fh
#import crop 

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from skimage import feature
from skimage.color import rgb2gray
from skimage import filters
from skimage import exposure
from skimage import morphology
from skimage import segmentation


import crop


#data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
data_dir=os.path.join(r'E:\OneDrive\WaterScope')


cfg=train_params(data_dir,base_db='db_categorized',curdb_dir='crop_problems')

save_dir=r'd:\DATA\WaterScope\tmp_problem'

#image_list=fh.imagelist_in_depth(cfg.base_imagedb_dir,level=2)
image_list=fh.imagelist_in_depth(cfg.curdb_dir,level=2)

#for image_file in image_list:
#        img = Image.open(image_file)
#        img_square=crop.crop(img,pad_rate=0.25,
#                             save_file=os.path.join(data_dir,'Images','tmp',os.path.basename(image_file)),
#                             category='dummy')


#file_names=[f for f in image_list]
class_names=[os.path.dirname(f).split('\\')[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':image_list,'Class name':class_names})

"""
Filter for classes

"""
#df_filtered=df_db[df_db['Class name']=='_Trash']
#df_filtered=df_db.copy()
#image_file=df_filtered.iloc[1]['Filename']
#image_file=r'e:\OneDrive\WaterScope\Images\crop_problems\0002678_DHM2.0.20151007.000..20151008T113053-0003.png'

#lengths=[]
char_size_dict={}
for i, image_file in enumerate(image_list):
#   i=6
    image_file=image_list[i]
    
    row=df_db.loc[df_db['Filename'] == image_file]
    
    if not row.empty:
        category=row['Class name'].values[0]
    
        try:    
            save_dir_cat=os.path.join(save_dir,category)
            if not os.path.exists(save_dir_cat):
                os.makedirs(save_dir_cat)
            save_file=os.path.join(save_dir_cat,os.path.basename(image_file))
        
        
            img = Image.open(image_file)
#            print(img.mode)
            print(i)
            if img.mode=='RGB':
                img.close()
                os.remove(image_file)
                continue
        
            img_rgb=img.convert('RGB')     # if png
             #   img_rgb=img
            img.close()

            
            im = np.asarray(img_rgb,dtype=np.uint8)
            
            
            bb,char_sizes, label_im, gray=crop.crop_segment(im)
            #im_gauss=filters.gaussian(im,2,multichannel=True) 
            
            # BENCH : im_mask = segmentation.felzenszwalb(im, scale=500, sigma=1,min_size=50)
            #im_mask = segmentation.felzenszwalb(im, scale=250, sigma=1.1,min_size=100)
#            
#            img_rgb.close()
#            gray=rgb2gray(im)
#            gray = filters.rank.median(gray, morphology.disk(3)).astype('float64')/255
#            
#        #    pixel_per_micron=crop.get_pixelsize(img)
#               
#            
#        #    print(str(pixel_per_micron)+' '+image_file)
#        #    lengths.append(pixel_per_micron)
#            
#            
##            contr=gray.max()-gray.min()
##            high_tsh=max(0.1,contr*0.32)
##            low_tsh=max(0.005,contr*0.01)
#         #   edges1 = feature.canny(gray, sigma=1, low_threshold=0.01, high_threshold=0.995, use_quantiles=True)
#         #   edges1 = feature.canny(gray, sigma=0.5, low_threshold=low_tsh, high_threshold=high_tsh, use_quantiles=False)
#        
#            im_gauss_1=filters.gaussian(gray,0.8)    
#            im_gauss_2=filters.gaussian(gray,1.5)   
#            im_diff=im_gauss_1-im_gauss_2
#            im_adj=exposure.rescale_intensity(gray)
#            
#            im_mask = segmentation.felzenszwalb(im_adj, scale=750, sigma=1,min_size=50)
        
#            im_mask=np.logical_and(im_mask,im_adj>0.2)
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))            
            
            ax1.imshow(gray)
            ax1.axis('off')  
            
            ax2.imshow(label_im)
            ax2.axis('off')    
            
            ax3.imshow(im)
            #ax1.axis('off')
       
            ax3.add_patch(patches.Rectangle(
                        (bb[1], bb[0]),   # (x,y)
                            bb[3]-bb[1],        # width
                            bb[2]-bb[0],       # height
                            fill=False))
         
            
            fig.savefig(save_file)
            plt.close('all')

        #    save_file=os.path.join(save_dir,os.path.basename(image_file))
        #    img_square, char_sizes=crop.crop(img,pad_rate=0.25,save_file=save_file,category='')
        except:
            print('error: '+image_file)

