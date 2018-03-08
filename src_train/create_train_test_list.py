# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:48:16 2017

@author: SzMike


Creates train and test lists from the images 
Each classes used have to have enough observations (min_obs)
creates type_dict
"""
"""
training_id='20180308'
"""
import csv
import pandas as pd
import os
import numpy as np
import collections

from src_train.train_config import train_params
#imp.reload(sys.modules['train_params'])

#==============================================================================
# SET THESE PARAMETERS!
#==============================================================================
curdb_dir='db_cropped_rot'
data_dir=os.path.join('/','home','mikesz','ownCloud','WaterScope')

#==============================================================================
# RUN CONFIG
#==============================================================================


cfg=train_params(data_dir,base_db='db_categorized',curdb_dir=curdb_dir,training_id=training_id)
typedict_file=os.path.join(cfg.train_dir,'type_dict.csv')


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

# do startified random split in the data
def get_stratified_train_test_inds(y,train_proportion=0.8):
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

    return np.where(train_inds)[0],np.where(test_inds)[0]



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

for label, class_name in enumerate(sorted(classes.unique())):
    type_dict[label]=class_name
    
# write type_dict
out = open(typedict_file, 'wt')
w = csv.DictWriter(out, delimiter=':', fieldnames=['label','type'])
w.writeheader()
for key, value in type_dict.items():
    w.writerow({'label' : key, 'type' : value})
out.close()
    
labels=[]
for cl in classes:
    labels.append(keysWithValue(type_dict,cl)[0])
    
    
"""
Spit to test and train sest
Using original image names
"""

orig_names=[os.path.basename(row['Filename']).split('__')[0] for i, row in df_filtered.iterrows()]
#import collections
#print([item for item, count in collections.Counter(orig_names).items() if count >4])
df_filtered['Origname']=orig_names

cats=[]
for name, group in df_filtered.groupby('Origname'):
    cats.append(group.iloc[0]['Class name'])
  
train_inds,test_inds = get_stratified_train_test_inds(cats, cfg.trainRatio)

train_inds_exp=[]
test_inds_exp=[]
for i, group in enumerate(df_filtered.groupby('Origname')):
    if i in train_inds:
        train_inds_exp=train_inds_exp+list(group[1].index)
    else:
        test_inds_exp=test_inds_exp+list(group[1].index)
 
np.random.shuffle(train_inds_exp)
np.random.shuffle(test_inds_exp)

"""
Label and size dataframes
"""
df_labeled=df_filtered[['Filename']].copy()
df_labeled['category']=labels
df_labeled.columns=['image','category']

df_sizes=df_filtered[['minl','maxl']].copy()


"""
Spit to test and train sest
"""


df_train_image=df_labeled.loc[train_inds_exp]
df_test_image=df_labeled.loc[test_inds_exp]
df_train_text=df_sizes.loc[train_inds_exp]
df_test_text=df_sizes.loc[test_inds_exp]

#orig_names=[os.path.basename(row['image']).split('__')[0] for i, row in df_train_image.iterrows()]
#df_train_image['Origname']=orig_names
#for name, group in df_train_image.groupby('Origname'):
#   if group.shape[0]!=4:
#       print(group)

"""
Do some stats
"""
num_classes=len(df_labeled['category'].value_counts())

classes_count_train=df_train_image['category'].value_counts()
print(len(df_train_image))
classes_count_test=df_test_image['category'].value_counts()
print(len(df_test_image))
# number of classes
print(num_classes)

"""
Write train and test list
"""
df_train_image.to_csv(cfg.train_image_list_file,sep=';',index=None)
df_test_image.to_csv(cfg.test_image_list_file,sep=';',index=None)
df_train_text.to_csv(cfg.train_text_list_file,sep=';',index=None)
df_test_text.to_csv(cfg.test_text_list_file,sep=';',index=None)
