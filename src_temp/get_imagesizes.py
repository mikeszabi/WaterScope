# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:48:01 2017

@author: picturio
"""
import imp
import sys
imp.reload(sys.modules['src_tools.image_helper'])
import src_tools.image_helper as ih

ms=[]
for i, image_file in enumerate(image_list):
#   i=1
    image_file=image_list[i]
    ms.append(max(ih.get_image_size(image_file)))