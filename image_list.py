# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:32:45 2017

@author: SzMike
"""


import os
import glob
import csv

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']

image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(r'f:\export', ext)))

with open(r'd:\DATA\WaterScope\image_list.txt', 'wt') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    for im in image_list_indir:
        wr.writerow([os.path.basename(im)])
