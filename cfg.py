# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:23:32 2017

@author: picturio
"""

import os

trainRatio=0.75
included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']


user='SzMike'
#user='picturio'
data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#data_dir=r'd:\DATA\WaterScope'


db_image_dir=os.path.join(data_dir,'db_images')
proc_image_dir=os.path.join(data_dir,'processed_images')
train_dir=os.path.join(data_dir,'Training')


db_file=os.path.join(db_image_dir,'Database.csv')

train_image_list_file=os.path.join(train_dir,'images_train.csv')
test_image_list_file=os.path.join(train_dir,'images_test.csv')
typedict_file=os.path.join(db_image_dir,'TypeDict.csv')
typedict_2_file=os.path.join(db_image_dir,'TypeDict_2.csv')
typedict_3_file=os.path.join(db_image_dir,'TypeDict_3.csv')

aliases_file=os.path.join(db_image_dir,'aliases.txt')
