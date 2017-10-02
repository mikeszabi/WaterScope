# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:06:04 2017

@author: SzMike
"""

from matplotlib import pyplot as plt
import glob

import classifications



import pandas as pd
import os
import shutil

import numpy as np
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL



user='SzMike'
imgSize=32
num_classes  = 2


data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
db_image_dir=os.path.join(data_dir,'db_images')
db_binsel_dir=os.path.join(data_dir,'db_binsel_images')
db_file=os.path.join(db_image_dir,'Database.csv')


user='picturio'
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#output_base_dir=r'd:\DATA\WaterScope'

train_dir=os.path.join(output_base_dir,'Training')

model_file=os.path.join(r'D:\Projects\WaterScope\model','cnn_model_binary.dnn')


df_db = pd.read_csv(db_file,delimiter=';')

type_dict={'Trash':'0','Object':'1'}

# LOAD MODEL

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']

image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(db_image_dir, ext)))

crop_map={}


cnn=classifications.cnn_classification(model_file)

for i, image_file in enumerate(image_list_indir):

    #image_file=os.path.join(r'C:\Users\SzMike\OneDrive\WaterScope\db_images\0044094_DHM2.0.20130101.Measurement20160926_161935..20161024T115654-0010.png')
    im = classifications.create_image(image_file,cropped=False)
    predicted_label, prob = cnn.classify(im)
    if predicted_label==1:
        shutil.copy(image_file,os.path.join(db_binsel_dir,os.path.basename(image_file)))