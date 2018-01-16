# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 08:14:28 2017

@author: picturio
"""
import os
import pandas as pd
from pandas.tools import plotting
import matplotlib.pyplot as plt

# run create_dbfile
size_file=os.path.join(cfg.base_imagedb_dir,'char_sizes.txt')

df_char_size=pd.read_csv(size_file,delimiter=';')

group_by_class=df_char_size.groupby('Class name')

fig, ax = plt.subplots()
df_char_size.boxplot('minl', by='Class name', ax=ax, figsize=(12, 8),rot=90, showfliers=False)

for class_name, ml in group_by_class['minl']:
    print(class_name+' : '+str(ml.mean()))

for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))
