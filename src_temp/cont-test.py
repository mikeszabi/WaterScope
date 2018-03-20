# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:23:24 2018

@author: SzMike
"""

import pandas as pd

csv_file=r'd:\DATA\WaterScope\testing\images_test.csv'

class_map_file=r'd:\DATA\WaterScope\testing\class_map.csv'

def read_merge_dict(class_map_file):
    merge_dict={}
    with open(cfg.merge_file, mode='r') as infile:
        reader = csv.reader(infile,delimiter=':')
        next(reader,None) # skip header
        for rows in reader:
            if rows:
                if rows[0][0]!='#':
                    merge_dict[rows[0]]=rows[1]
    return merge_dict

cur_dir=r'd:\DATA\WaterScope\testing'
for root, dirs, files in walklevel(measure_dir, level=1):
    dir_struct=root.split(os.path.sep)
    print(root)