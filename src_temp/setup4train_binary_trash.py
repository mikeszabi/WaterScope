# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:48:16 2017

@author: SzMike


Creates train and test lists from the images
Each class es used has to have enough obsercations (min_obs)
"""


import glob
import csv
from collections import Counter
import random
import math
import pandas as pd
import os
import src_train.train_config as cfg
#import numpy as np


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)


"""
Read data description file
"""

df_db = pd.read_csv(db_file,delimiter=';')

"""
Filter data description file
Select classes to process
"""

df_filtered=df_db[(df_db['Class quality']=='highclass')]

samples={}


image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(db_image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
#   i=1
    image_file=image_list_indir[i]
    
    row=df_filtered.loc[df_filtered['Filename'] == os.path.basename(image_file)]
    
    if not row.empty:
        samples[image_file]=1
    
image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(trash_image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
#   i=1
    image_file=image_list_indir[i]
    
    samples[image_file]=0

sampleCount=Counter(samples.values())

print('Number of classes : '+str(len(sampleCount)))        
print('Sample size of smaller class : '+str(sampleCount[0]))

# CREATE TEST AND TRAIN LIST USING RANDOM SPLIT
i=0
testProds = {}
trainProds = {}

train_image_list_file=os.path.join(train_dir,'images_train_binary.csv')
test_image_list_file=os.path.join(train_dir,'images_test_binary.csv')

for cat, count in sampleCount.items():
    catProds=keysWithValue(samples,cat)
    random.shuffle(catProds)
    splitInd=int(math.ceil(trainRatio*len(catProds)))
    trainItems=catProds[:splitInd]
    testItems=catProds[splitInd:]
    for item in testItems:
        testProds[item]=cat
    for item in trainItems:
        trainProds[item]=cat

out = open(train_image_list_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['image','category'])
w.writeheader()
for key, value in trainProds.items():
    w.writerow({'image' : key, 'category' : value})
out.close()

out = open(test_image_list_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['image','category'])
w.writeheader()
for key, value in testProds.items():
    w.writerow({'image' : key, 'category' : value})
out.close()

