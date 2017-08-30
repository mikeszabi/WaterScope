# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:57:02 2017

@author: picturio
"""

import os
import glob

def walklevel(root_dir, level=1):
    root_dir = root_dir.rstrip(os.path.sep)
    assert os.path.isdir(root_dir)
    num_sep = root_dir.count(os.path.sep)
    for root, dirs, files in os.walk(root_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            
def imagelist_in_depth(image_dir,level=1):
    image_list_indir=[]
    included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
    image_list_indir = []
    for root, dirs, files in walklevel(image_dir, level=level):
        for ext in included_extenstions:
            image_list_indir.extend(glob.glob(os.path.join(root, ext)))
    return image_list_indir

def check_folder(folder='.',create=True):
    if os.path.exists(folder):
        return True
    else:
        if create:
            os.makedirs(folder)
            return True
        else:
            return False