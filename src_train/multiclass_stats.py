# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:16:28 2017

@author: SzMike
"""

import numpy as np
# Calculate statistical measures for multiclass classification
# 1 vs. all single class approach
# in cont_table rows are actual labels, cols are predictions!     
def multiclass_statistics(cont_table,macro=False):
           
    beta=1 # used in fscore
    stats={}
    
    num_classes=cont_table.shape[0]
    n_obs=cont_table.sum().sum()
    
    # Allocate memory
    tp=[None]*num_classes # correct inclass lassification
    tn=[None]*num_classes # correct outclass classification - 1 vs. all
    fp=[None]*num_classes # incorrect inclass classification
    fn=[None]*num_classes # incorrect outclass classification
    
    # Calculate classification rates        
    tp=[cont_table.iloc[i,i] for i in range(0,num_classes)] # correctly identified - diagonal values
    fp=[-tp[i]+sum(cont_table.iloc[i,:]) for i in range(0,num_classes)] # incorrectly identified class members - col. values except diagonal
    fn=[-tp[i]+sum(cont_table.iloc[:,i]) for i in range(0,num_classes)] # incorrectly identified non-class members - row values except diagonal
    tn=[n_obs-fp[i]-fn[i]-tp[i] for i in range(0,num_classes)] # correctly identified non-class members - not in row or col.
    
    # calculate statistics
    # recall - same as sensitivity
    
    if macro:
        # 1., MACRO - all classes equally weighted
        stats['precision']=sum([(tp[i])/(tp[i]+fp[i]) for i in range(0,num_classes)])/num_classes
        stats['recall']=sum([(tp[i])/(tp[i]+fn[i]) for i in range(0,num_classes)])/num_classes
        stats['specificity']=sum([(tn[i])/(tn[i]+fp[i]) for i in range(0,num_classes)])/num_classes
        stats['avg_accuracy']=sum([(tp[i]+tn[i])/(tp[i]+fn[i]+fp[i]+tn[i]) for i in range(0,num_classes)])/num_classes
    else:
        # 2., MICRO - larger classes have more weight
        stats['precision']=sum([(tp[i]) for i in range(0,num_classes)])/sum([(tp[i]+fp[i]) for i in range(0,num_classes)])
        stats['recall']=sum([(tp[i]) for i in range(0,num_classes)])/sum([(tp[i]+fn[i]) for i in range(0,num_classes)])
        stats['specificity']=sum([(tn[i]) for i in range(0,num_classes)])/sum([(tn[i]+fp[i]) for i in range(0,num_classes)])
        stats['avg_accuracy']=sum([(tp[i]) for i in range(0,num_classes)])/n_obs
    
    stats['fscore']=(np.square(beta)+1)*stats['precision']*stats['recall']/(np.square(beta)*stats['precision']+stats['recall'])

    print(stats)
    return stats