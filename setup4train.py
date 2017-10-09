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
from cfg import included_extensions, proc_image_dir, typedict_2_file, aliases_file, typedict_3_file, trainRatio, train_image_list_file, test_image_list_file

#import pandas as pd
#import numpy as np


#user='picturio'
#data_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
##data_dir=r'd:\DATA\WaterScope'

def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

#trainRatio=0.75


#included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
#
#db_image_dir=os.path.join(data_dir,'db_images')
#proc_image_dir=os.path.join(data_dir,'cropped_highclass_20170823')
#
#train_dir=os.path.join(data_dir,'Training')
#train_image_list_file=os.path.join(train_dir,'images_train.csv')
#test_image_list_file=os.path.join(train_dir,'images_test.csv')
##typedict_file=os.path.join(orig_dir,'TypeDict.csv')
#typedict_2_file=os.path.join(db_image_dir,'TypeDict_2.csv')
#typedict_3_file=os.path.join(db_image_dir,'TypeDict_3.csv')

#db_file=os.path.join(data_dir,'Database.csv')

# read type dict
#type_dict={}
#reader =csv.DictReader(open(typedict_file, 'rt'), delimiter=';')
#for row in reader:
#    type_dict[row['type']]=row['label']

image_list_indir = []
for ext in included_extensions:
    image_list_indir.extend(glob.glob(os.path.join(proc_image_dir, ext)))


#df = pd.read_csv(db_file,delimiter=';')
#
#types=pd.Series(df['Class name'])
#
#print(types.value_counts())

samples_all = {}
for i, image_file in enumerate(image_list_indir):
#    i=10
    image_file=image_list_indir[i]
    file_name=os.path.basename(image_file)
    #print(str(i)+' : '+os.path.basename(file_name))
    #label=df.loc[df['Filename'] == os.path.basename(image_file)]
    #category=label['Class name'].values[0]
    category=file_name.split('__')[0]
    samples_all[image_file]=category


sampleCount=Counter(samples_all.values())

# remove samples with low counts
samples_enough = {k: v for k, v in samples_all.items() if sampleCount[v]>50*4}

categories=sorted(Counter(samples_enough.values()))

print('Number of classes : '+str(len(categories)))

# categories in alphabetical order 
type_dict_2 = {}
for label, type in enumerate(categories):
    type_dict_2[type]=label
    

out = open(typedict_2_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['type','label'])
w.writeheader()
for key, value in type_dict_2.items():
    w.writerow({'type' : key, 'label' : value})
out.close()

# new labels
samples={}

for k,v in samples_enough.items():
    file_name=os.path.basename(k)
    samples[k]=type_dict_2[v]


sampleCount=Counter(samples.values())

print('Number of classes : '+str(len(sampleCount)))


# add group aliases
aliases={}
reader =csv.DictReader(open(aliases_file, 'rt'), delimiter=':')
for row in reader:
    aliases[row['alias']]=row['classes']
    
types_3={}
for cat in type_dict_2.keys():
    found=False
    for alias,group in aliases.items():
        for tp in group.split(','):
            if cat==tp:
                types_3[cat]=alias
                found=True
                break
        if found:
            break
    if not found:
        types_3[cat]=cat
              
type_dict_3={}
for i,tp in enumerate(sorted(set(types_3.values()), key=lambda v: v.upper())):
    type_dict_3[tp]=i
               
out = open(typedict_3_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['type','label'])
w.writeheader()
for key, value in type_dict_3.items():
    w.writerow({'type' : key, 'label' : value})
out.close()
 
samples={}

for k,v in samples_enough.items():
    file_name=os.path.basename(k)
    samples[k]=type_dict_3[types_3[v]]

sampleCount=Counter(samples.values())

print('Number of classes : '+str(len(sampleCount)))        

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

