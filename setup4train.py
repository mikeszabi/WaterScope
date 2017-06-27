# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:48:16 2017

@author: SzMike
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:40:14 2016

@author: picturio
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:18:47 2016

@author: SzMike
"""
import glob
import csv
from collections import Counter
import random
import math
import os


#import pandas as pd
#import numpy as np


user='picturio'
data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#data_dir=r'd:\DATA\WaterScope'

def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

trainRatio=0.75


included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']

image_dir=os.path.join(data_dir,'cropped')
train_dir=os.path.join(data_dir,'Training')
train_image_list_file=os.path.join(train_dir,'images_train.csv')
test_image_list_file=os.path.join(train_dir,'images_test.csv')
typedict_file=os.path.join(data_dir,'TypeDict.csv')


#db_file=os.path.join(data_dir,'Database.csv')

# read type dict
type_dict={}
reader =csv.DictReader(open(typedict_file, 'rt'), delimiter=';')
for row in reader:
    type_dict[row['type']]=row['label']

image_list_indir = []
for ext in included_extenstions:
    image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))


#df = pd.read_csv(db_file,delimiter=';')
#
#types=pd.Series(df['Class name'])
#
#print(types.value_counts())

samples = {}
for i, image_file in enumerate(image_list_indir):
#    i=10
    image_file=image_list_indir[i]
    file_name=os.path.basename(image_file)
    #print(str(i)+' : '+os.path.basename(file_name))
    #label=df.loc[df['Filename'] == os.path.basename(image_file)]
    #category=label['Class name'].values[0]
    category=file_name.split('__')[0]
    samples[image_file]=type_dict[category]


sampleCount=Counter(samples.values())

# CREATE TEST AND TRAIN LIST USING RANDOM SPLIT
i=0
testProds = {}
trainProds = {}

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

