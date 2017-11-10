# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:48:16 2017

@author: SzMike


Creates train and test lists from the images 
Each class es used has to have enough obsercations (min_obs)
"""


import csv
import pandas as pd
import os
import numpy as np
import collections

from src_train.train_config import train_params
#imp.reload(sys.modules['train_params'])

# do startified random split in the data
def get_stratified_train_test_inds(y,train_proportion=0.7):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds



#data_dir=os.path.join(r'E:\OneDrive\WaterScope')

data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
cfg=train_params(data_dir,crop=True,training_id='20171110')


"""
Read data description file
"""

df_db = pd.read_csv(cfg.db_file,delimiter=';')

"""
Select classes to process
"""

df_filtered=df_db.copy()


"""
Set numeric class labels
"""
classes=df_filtered['Class name']
type_dict=collections.OrderedDict()

for label, class_name in enumerate(classes.unique()):
    type_dict[class_name]=label
    
# write type_dict
typedict_file=os.path.join(cfg.train_dir,'type_dict.csv')
out = open(typedict_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['type','label'])
w.writeheader()
for key, value in type_dict.items():
    w.writerow({'type' : key, 'label' : value})
out.close()
    
labels=[]
for cl in classes:
    labels.append(type_dict[cl])  
    
df_labeled=df_filtered[['Filename']]
df_labeled['category']=labels
df_labeled.columns=['image','category']


"""
Spit to test and train sest
"""
train_inds,test_inds = get_stratified_train_test_inds(df_labeled['category'], cfg.trainRatio)
df_train=df_labeled[train_inds] 
df_test=df_labeled[test_inds] 

"""
Do some stats
"""
classes_count_train=df_train['category'].value_counts()
print(len(df_train))
classes_count_test=df_train['category'].value_counts()
print(len(df_test))
# number of classes
print(len(df_labeled['category'].value_counts()))

"""
Write train and test list
"""
df_train.to_csv(cfg.train_image_list_file,sep=';',index=None)
df_test.to_csv(cfg.test_image_list_file,sep=';',index=None)

