# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:34:22 2017

@author: picturio
"""
from cntk.io import MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDef, StreamDefs, transforms

#
# Define the reader for both training and evaluation action.
#
image_height = 64
image_width  = 64
num_channels = 3
num_classes  = 32

def create_reader(map_file, mean_file, train, image_height=64, image_width=64, num_channels=3, num_classes=32):
  
    # transformation pipeline for the features has crop only when training

    trs = []
    if train:
        trs += [
            transforms.crop(crop_type='randomside', side_ratio=0, jitter_type='none') # Horizontal flip enabled
        ]
    trs += [
        transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        transforms.mean(mean_file)
    ]
    # deserializer
    image_source=ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=trs), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    ))
    return MinibatchSource(image_source)

# https://cntk.ai/pythondocs/Manual_How_to_feed_data.html
# https://docs.microsoft.com/en-us/cognitive-toolkit/brainscript-cntktextformat-reader
def create_reader_with_size(map_file_image, map_file_size, mean_file_image, train, image_height=64, image_width=64, num_channels=3, num_classes=32):
  
    # transformation pipeline for the features has jitter/crop only when training
    # https://docs.microsoft.com/en-us/python/api/cntk.io.transforms?view=cntk-py-2.2
    trs = []
    if train:
        trs += [
            transforms.crop(crop_type='randomside', side_ratio=0, jitter_type='none') # Horizontal flip enabled
        ]
    trs += [
        transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        transforms.mean(mean_file_image)
    ]
    # deserializer
    image_source=ImageDeserializer(map_file_image, StreamDefs(
        features = StreamDef(field='image', transforms=trs), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    ))
    size_source = CTFDeserializer(map_file_size, StreamDefs(
        minl = StreamDef(field='minl', shape=1,is_sparse=False),
        maxl = StreamDef(field='maxl', shape=1,is_sparse=False)
    ))
        
    return MinibatchSource(image_source,size_source)

#https://docs.microsoft.com/en-us/cognitive-toolkit/brainscript-cntktextformat-reader
#|A 0 1 2 3 4 |# a CTF comment
