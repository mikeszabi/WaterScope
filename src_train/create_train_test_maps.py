# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:29:36 2016

@author: SzMike
"""

#training_id='20171126-Gray'

# https://github.com/Microsoft/CNTK/blob/v2.0.beta6.0/Tutorials/CNTK_201A_CIFAR-10_DataLoader.ipynb

import pandas as pd
import numpy as np
import os
import xml.etree.cElementTree as et
import xml.dom.minidom
import csv

from src_train.train_config import train_params
#%matplotlib inline

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

##

train_map_o=os.path.join(cfg.train_dir,'train_map.txt')
test_map_o=os.path.join(cfg.train_dir,'test_map.txt')
#train_regr_labels=os.path.join(train_dir,'train_regrLabels.txt')
#data_mean_file=os.path.join(cfg.train_dir,'data_mean.xml')


def balanceMap(mapfile,min_count=100,max_count=200):
    df = pd.read_csv(mapfile,delimiter='\t',names=('image','label'))
    label_grouping = df.groupby('label')

    df_enrich = [] #pd.DataFrame({'image' : [], 'label' : []})
    e_groups = []
    for count,label in label_grouping:
        N=min(max(len(label),min_count),max_count)
        e_groups.append(label.sample(N,replace=True))    
    df_enrich=pd.concat(e_groups)    
    # random shuffle
    df_enrich=df_enrich.sample(frac=1)  
    head, tail=os.path.splitext(mapfile)
    e_mapfile=head+'_e'+str(min_count)+'_'+str(max_count)+tail
    df_enrich.to_csv(e_mapfile,header=False,sep='\t')
    
    return e_mapfile
 

def saveImage_row(fname, label,mapFile):

 
    mapFile.write("%s\t%d\n" % (fname, label))
    
def saveText_row(maxl,minl,mapFile):

    mapFile.write('|size '+str(maxl)+' '+str(minl)+'\n')

#    
#def saveMean(fname, data):
#    root = et.Element('opencv_storage')
#    et.SubElement(root, 'Channel').text = str(num_channels)
#    et.SubElement(root, 'Row').text = str(image_height)
#    et.SubElement(root, 'Col').text = str(image_width)
#    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
#    et.SubElement(meanImg, 'rows').text = '1'
#    et.SubElement(meanImg, 'cols').text = str(numFeature)
#    et.SubElement(meanImg, 'dt').text = 'f'
#    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (numFeature))])
#
#    tree = et.ElementTree(root)
#    tree.write(fname)
#    x = xml.dom.minidom.parse(fname)
#    with open(fname, 'w') as f:
#        f.write(x.toprettyxml(indent = '  '))

def saveImages(filename, train_dir,map_type='train'):
    
    if map_type=='train':
        map_file=os.path.join(train_dir,'train_map_image.txt')
    else:
        map_file=os.path.join(train_dir,'test_map_image.txt')
    
    reader =csv.DictReader(open(filename, 'rt'), delimiter=';')

    with open(map_file, 'w') as mapFile:
        for row in reader:
            label=int(row['category'])
            fname = row['image']
           
            saveImage_row(fname, label, mapFile)
    
#    if map_type=='train' :
#        dataMean = 128*np.ones((num_channels, image_height, image_width))
#        saveMean(os.path.join(train_dir,'data_mean.xml'), dataMean)
        
def saveTexts(filename, train_dir,map_type='train'):
    
    if map_type=='train':
        map_file=os.path.join(train_dir,'train_map_text.txt')
    else:
        map_file=os.path.join(train_dir,'test_map_text.txt')
    
    reader =csv.DictReader(open(filename, 'rt'), delimiter=';')

    with open(map_file, 'w') as mapFile:
        for row in reader:
            maxl=int(row['maxl'])
            minl=int(row['minl'])
           
            saveText_row(maxl, minl, mapFile)

print ('Converting train data to png images...')
saveImages(cfg.train_image_list_file,cfg.train_dir,map_type='train')
if os.path.exists(cfg.train_text_list_file):
    saveTexts(cfg.train_text_list_file,cfg.train_dir,map_type='train')

print ('Done.')
print ('Converting test data to png images...')
saveImages(cfg.test_image_list_file,cfg.train_dir,map_type='test')
if os.path.exists(cfg.test_text_list_file):
    saveTexts(cfg.test_text_list_file,cfg.train_dir,map_type='test')

print ('Done.')

# enrich!
#train_map=balanceMap(train_map_o,min_count=600, max_count=3000)
#test_map=balanceMap(test_map_o,min_count=200, max_count=1000)