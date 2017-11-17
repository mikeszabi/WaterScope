# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 08:14:28 2017

@author: picturio
"""

from pandas.tools import plotting


group_by_class=pd_char_size.groupby('Class name')


pd_char_size.boxplot('minl', by='Class name', figsize=(12, 8),rot=90)


for class_name, ml in group_by_class('maxl'):
    print({class_name,ml.mean()})

for gender, value in groupby_gender['VIQ']:
...     print((gender, value.mean()))
pd_char_size