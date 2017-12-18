# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:23:32 2017

@author: picturio
"""

import os
from src_tools.file_helper import supermakedirs

class train_params:
    trainRatio=0.75
    included_extensions = ['*.jpg', '*.bmp', '*.png', '*.gif']
    
    def __init__(self,data_dir='C:',curdb_dir='cropped_gray_images',training_id=''):
        if os.path.exists(data_dir):
            self.base_imagedb_dir=os.path.join(data_dir,'Images','db_korea')
            self.curdb_dir=os.path.join(data_dir,'Images',curdb_dir)
                
            self.train_dir=os.path.join(data_dir,'Training_'+training_id)    
            self.db_file=os.path.join(self.train_dir,'Database.csv')
            self.classnames_file=os.path.join(self.curdb_dir,'class_count.csv')
            
            self.merge_file=os.path.join(self.train_dir,'class_map.csv')
            
            self.train_dir=os.path.join(data_dir,'Training_'+training_id)
            supermakedirs(self.train_dir,0o777)
            self.train_log_dir=os.path.join(self.train_dir,'log')
            supermakedirs(self.train_log_dir,0o777)

            self.train_image_list_file=os.path.join(self.train_dir,'images_train.csv')
            self.test_image_list_file=os.path.join(self.train_dir,'images_test.csv')
            self.train_text_list_file=os.path.join(self.train_dir,'text_train.csv')
            self.test_text_list_file=os.path.join(self.train_dir,'text_test.csv')
        else:
            print('data dir does not exist')

            