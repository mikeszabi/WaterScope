# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:23:32 2017

@author: picturio
"""

import os


class train_params:
    trainRatio=0.75
    included_extensions = ['*.jpg', '*.bmp', '*.png', '*.gif']
    
    def __init__(self,data_dir='C:'):
        if os.path.exists(data_dir):
            self.imagedb_dir=os.path.join(data_dir,'Images','db_categorized')
            self.imagecrop_dir=os.path.join(data_dir,'Images','cropped_images')

            self.db_file=os.path.join(self.imagedb_dir,'Database.csv')
            self.classnames_file=os.path.join(self.imagedb_dir,'AllClasses.csv')
            self.merge_file=os.path.join(self.imagedb_dir,'ClassMerge.csv')
            
            self.train_dir=os.path.join(data_dir,'Training')

            self.train_image_list_file=os.path.join(self.train_dir,'images_train.csv')
            self.test_image_list_file=os.path.join(self.train_dir,'images_test.csv')
        else:
            print('data dir does not exist')

            



##user='SzMike'
#user='picturio'
#data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#
#
#imagedb_dir=os.path.join(data_dir,'Images','db_categorized')
#
#db_file=os.path.join(imagedb_dir,'Database.csv')
#classnames_file=os.path.join(imagedb_dir,'AllClasses.csv')
#merge_file=os.path.join(imagedb_dir,'ClassMerge.csv')
#
#imagecrop_dir=os.path.join(data_dir,'Images','cropped_images')
##trash_image_dir=os.path.join(data_dir,'Micro Supertrash')
#
##proc_image_dir=os.path.join(data_dir,'processed_images_highclass')
#train_dir=os.path.join(data_dir,'Training')
#
#train_image_list_file=os.path.join(train_dir,'images_train.csv')
#test_image_list_file=os.path.join(train_dir,'images_test.csv')
##typedict_file=os.path.join(db_image_dir,'TypeDict.csv')
##typedict_2_file=os.path.join(db_image_dir,'TypeDict_2.csv')
##typedict_3_file=os.path.join(db_image_dir,'TypeDict_3.csv')
##
##aliases_file=os.path.join(db_image_dir,'aliases.txt')
