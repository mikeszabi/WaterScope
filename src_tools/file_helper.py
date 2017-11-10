# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:57:02 2017

@author: picturio
"""

import os
import glob
import re
import pandas as pd

def walklevel(root_dir, level=1):
    root_dir = root_dir.rstrip(os.path.sep)
    assert os.path.isdir(root_dir)
    num_sep = root_dir.count(os.path.sep)
    for root, dirs, files in os.walk(root_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            
def imagelist_in_depth(image_dir,level=1):
    image_list_indir=[]
    included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
    for root, dirs, files in walklevel(image_dir, level=level):
        for ext in included_extenstions:
            image_list_indir.extend(glob.glob(os.path.join(root, ext)))
    return image_list_indir

def images2process_list_in_depth(measure_dir,file2check1=['settings-settings.xml','control.log'],file2check2=['(Measure).*(.xml)'],level=3):
    df_images2process=pd.DataFrame(columns=['root','dir1','dir2','image_file'])
    i_row=-1
    included_extenstions = ['png']
    for root, dirs, files in walklevel(measure_dir, level=level):
        dir_struct=root.split('\\')
        if len(dir_struct)==measure_dir.count(os.path.sep)+level:
             # root: ..\\Measure\\DATE\\MeasureID
            if files:
                # not empty dir
                if len(set(files).intersection(file2check1))==len(file2check1): 
                    regex=re.compile(file2check2[0])
                    fc=[f for f in files if regex.search(f)]
                    if not fc:
                        # not classifed yet
                        #print(files)
                        image_files=[f for f in files if set(included_extenstions).intersection(f.split('.'))]
                        # ToDo: check date
                        if image_files:
                            for image_file in image_files:
                                i_row+=1
                                if level>2:
                                    df_images2process.loc[i_row]=[root,dir_struct[-2],dir_struct[-1],image_file]
                                else:
                                    df_images2process.loc[i_row]=[root,'','',image_file]
    
    return df_images2process

def read_log(file_name):
    # log is "=" separated
    log_df=pd.read_csv(file_name,sep='=')
    log_dict={row[0][0:-1]:row[1][1:] for i, row in log_df.iterrows()}

    return log_dict

def dirlist_onelevel(cur_dir,level=1):
    dir_list = [f.path.split('\\')[-1] for f in os.scandir(cur_dir) if f.is_dir() ] 
    return dir_list

def check_folder(folder='.',create=True):
    if os.path.exists(folder):
        return True
    else:
        if create:
            os.makedirs(folder)
            return True
        else:
            return False
        
def supermakedirs(path, mode):
    if not path or os.path.exists(path):
        return []
    (head, tail) = os.path.split(path)
    res = supermakedirs(head, mode)
    os.mkdir(path)
    os.chmod(path, mode)
    res += [path]
    return res