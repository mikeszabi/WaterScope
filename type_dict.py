# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 20:15:19 2017

@author: SzMike
"""

import pandas as pd
import os
import csv

user='picturio'
data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#data_dir=r'd:\DATA\WaterScope'


db_file=os.path.join(data_dir,'Database.csv')
typedict_file=os.path.join(data_dir,'TypeDict.csv')

df = pd.read_csv(db_file,delimiter=';')

types=pd.Series(df['Class name'])

type_dict={}
for i, itype in enumerate(types.unique()):
    type_dict[itype]=i
    
out = open(typedict_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['type','label'])
w.writeheader()
for key, value in type_dict.items():
    w.writerow({'type' : key, 'label' : value})
out.close()