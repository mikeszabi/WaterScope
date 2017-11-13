# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:17:31 2017

@author: SzMike
"""
import os
import re
import xml.etree.ElementTree as ET
from src_tools.file_helper import walklevel


tree = ET.parse('cfg_SU_classifier.xml')
root_dir = tree.find('folders/root').text
measure_dir = os.path.join(root_dir,tree.find('folders/measurement').text)

measure_file = tree.find('files/measure').text


regex=re.compile(measure_file)
for root, dirs, files in walklevel(measure_dir, level=3):
    fc=[f for f in files if regex.search(f)]
    for f in fc:
        file2delete=os.path.join(root, f)
        print('deleting: '+file2delete)
        os.remove(file2delete)