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

# https://github.com/Microsoft/CNTK/blob/v2.0.beta6.0/Tutorials/CNTK_201A_CIFAR-10_DataLoader.ipynb

import warnings
import pandas as pd
import numpy as np
import os
import xml.etree.cElementTree as et
import xml.dom.minidom
import csv
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
from skimage.transform import resize
from skimage import img_as_ubyte
from PIL import Image
from cfg import *

#%matplotlib inline

user='picturio'
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WaterScope')
#output_base_dir=r'd:\DATA\WaterScope'


# Config matplotlib for inline plotting

imgSize=32
nCh=3
numFeature = imgSize * imgSize * 3
#num_classes  = 33
#
#label_base=np.zeros(num_classes)


train_image_list_file=os.path.join(train_dir,'images_train_binary.csv')
test_image_list_file=os.path.join(train_dir,'images_test_binary.csv')


#train_img_directory = os.path.join(train_dir,'Train')
#test_img_directory = os.path.join(train_dir,'Test')

train_map_o=os.path.join(train_dir,'train_map_binary.txt')
test_map_o=os.path.join(train_dir,'test_map_binary.txt')

#train_regr_labels=os.path.join(train_dir,'train_regrLabels.txt')

data_mean_file=os.path.join(train_dir,'data_mean_binary.xml')

def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)


def balanceMap(mapfile,min_count=100,max_count=500):
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
 

def writeMap(fname, pixData, label, mapFile,  **key_parms):

    if ('mean' in key_parms):
        key_parms['mean'] += pixData

#    if pad > 0:
#        pixData = np.pad(pixData, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=128) 

#    data=np.transpose(pixData, (1, 2, 0))  
#    io.imsave(fname,data)
 
    mapFile.write("%s\t%d\n" % (fname, label))
    
#    # compute per channel mean and store for regression example
#    channelMean = np.mean(pixData, axis=(1,2))
#    regrFile.write("|regrLabels\t%f\t%f\t%f\n" % (channelMean[0]/255.0, channelMean[1]/255.0, channelMean[2]/255.0))
#    
def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imgSize)
    et.SubElement(root, 'Col').text = str(imgSize)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imgSize * imgSize * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (imgSize * imgSize * 3))])

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))

def saveTrainImages(filename, train_dir):
#    if not os.path.exists(os.path.join(train_dir,'Train')):
#        os.makedirs(os.path.join(train_dir,'Train'))
    data = {}
    dataMean = np.zeros((3, imgSize, imgSize)) # mean is in CHW format.
    
# Szabi code
    reader =csv.DictReader(open(filename, 'rt'), delimiter=';')
    prods = {}

    with open(os.path.join(train_dir,'train_map_binary.txt'), 'w') as mapFile:
        for row in reader:
            label=int(row['category'])
#                lab=label_base.copy()
#                lab[label]=1
            prods[row['image']]=row['category']
# read image file
# create data sequence RRR GGG BBB
            fname = row['image']
            img=Image.open(fname)
            if img.format=='PNG':
                bg = img.convert('RGB')
            else:
                bg=img
            im  = np.array(bg, dtype=np.uint8)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = img_as_ubyte(resize(im, (imgSize,imgSize), order=1))
            #fname = os.path.join(train_dir,'Train',os.path.basename(row['image'])) # .decode('utf-8')  -solves unicode problem
            data=np.transpose(data, (2, 0, 1)) # CHW format.
            writeMap(fname, data, label, mapFile, mean=dataMean)

#    dataMean = dataMean / len(prods)
    dataMean = 128*np.ones((3, imgSize, imgSize))
    saveMean(os.path.join(train_dir,'data_mean_binary.xml'), dataMean)

def saveTestImages(filename, train_dir):
#    if not os.path.exists(os.path.join(train_dir,'Test')):
#        os.makedirs(os.path.join(train_dir,'Test'))
        
    # Szabi code
    reader =csv.DictReader(open(filename, 'rt'), delimiter=';')
    prods = {}

    with open(os.path.join(train_dir,'test_map_binary.txt'), 'w') as mapFile:
        for row in reader:
            prods[row['image']]=row['category']
# read image file
# create data sequence RRR GGG BBB

            fname = row['image']
            im = io.imread(fname)                  
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = img_as_ubyte(resize(im, (imgSize,imgSize), order=1))
            #fname = os.path.join(train_dir,'Test',os.path.basename(row['image'])) # .decode('utf-8')  -solves unicode problem
            data=np.transpose(data, (2, 0, 1)) # CHW format.
            writeMap(fname, data, int(row['category']), mapFile)


print ('Converting train data to png images...')
saveTrainImages(train_image_list_file,train_dir)
print ('Done.')
print ('Converting test data to png images...')
saveTestImages(test_image_list_file,train_dir)
print ('Done.')

# enrich!
train_map=balanceMap(train_map_o,min_count=0, max_count=10000)
test_map=balanceMap(test_map_o,min_count=0, max_count=2500)
