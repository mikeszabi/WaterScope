# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:13:49 2017

@author: SzMike
"""

res_file=r'd:\Projects\WaterScope\work\Measurement\20170712\002\classification_result.csv'

df=pd.read_csv(res_file) 

df.predicted_type.unique()

for c in df.predicted_type.value_counts():
    print(c)
    
df.predicted_type.value_counts()['Schroederia']

log_file=r'd:\Projects\WaterScope\work\Measurement\20170712\002\control.log'
df=pd.read_csv(log_file,sep='=',header=None) 


dd={row[0][0:-1]:row[1][1:] for i, row in df.iterrows()}
    print(row)