# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:59:11 2017

@author: SzMike
"""

import tkinter as tk
from tkinter.filedialog import askdirectory
import csv
import time
import os
import shutil
import xml.etree.ElementTree as ET
from src_tools.file_helper import imagelist_in_depth, images2process_list_in_depth, check_folder
import classifications
import pandas as pd

#import sys
#import logging

# Logging setup
#log_file='progress.log'
#logging.basicConfig(filename=log_file,level=logging.DEBUG)

def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

class params:
    
    def __init__(self):
        # reading parameters
        tree = ET.parse('cfg_2steps.xml')
        self.dirs={}
        self.files={}
        self.dirs['root'] = tree.find('folders/root').text 
        self.dirs['measurement'] = tree.find('folders/measurement').text 
        self.dirs['classification'] = tree.find('folders/classification').text 
        self.dirs['results'] = tree.find('folders/results').text 
        self.files['control'] = tree.find('files/control').text 
        self.files['measure'] = tree.find('files/measure').text
        self.files['model_trash'] = os.path.join(tree.find('folders/model').text,
                                          tree.find('files/model_trash').text)
        self.files['model_taxon'] = os.path.join(tree.find('folders/model').text,
                                          tree.find('files/model_taxon').text)
        self.files['typedict'] = os.path.join(tree.find('folders/model').text,
                                          tree.find('files/typedict').text)
        
        self.type_dict_taxon=self.get_typedict(self.files['typedict'])
        
        self.type_dict_trash={'Recycle bin':'0','Object':'1'}
        
    def get_typedict(self,typedict_file):
        type_dict={}
        if os.path.isfile(typedict_file):
            reader =csv.DictReader(open(typedict_file, 'rt'), delimiter=';')
            for row in reader:
                type_dict[row['type']]=row['label']
            print('typeDict loaded')
        else:
            for i in range(100):
                type_dict[str(i)]=str(i)
        return type_dict
                                     

class Application(tk.Frame):
    running=False
    cur_folder='.'
       
    def __init__(self, master=None):
        self.params=params()

        tk.Frame.__init__(self, master, background="green")
        self.pack(fill="both", expand=True)
        self.createWidgets()
        
        
        self.cnn_1=classifications.cnn_classification(self.params.files['model_trash'])
        print('load model '+self.params.files['model_trash'])
        self.cnn_2=classifications.cnn_classification(self.params.files['model_taxon'])
        print('load model '+self.params.files['model_taxon'])
        
#        # check result folder
#        check_folder(folder=os.path.join(self.params.dirs['root'],self.params.dirs['classification']),create=True)
#        check_folder(folder=os.path.join(self.params.dirs['root'],self.params.dirs['classification'],self.params.dirs['results']),create=True)
#        print('result folders are checked and created')

     
    def createWidgets(self)   :        
        # define widgets
        scrollbar = tk.Scrollbar(self)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text=tk.Text(self,height=10,width=50, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.text.pack(expand=True, fill='both', side=tk.BOTTOM)
        scrollbar.config(command=self.text.yview)

        separator = tk.Frame(height=2, bd=1, relief=tk.SUNKEN)
        separator.pack(fill=tk.X, padx=5, pady=5)
        
        self.dir_label = tk.Label(self, text='root'+' : '+self.params.dirs['root'])
        self.dir_label.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.date_entry = tk.Entry(self, bg='yellow')
        self.date_entry.pack(fill=tk.X, side=tk.BOTTOM)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d"))
        
        test_button = tk.Button(separator, text="TEST", command=self.test, width=10)
        test_button.pack(side=tk.RIGHT)
        
        start_button = tk.Button(separator, text="START", command=self.start, width=10)
        start_button.pack(side=tk.LEFT)
        
        stop_button = tk.Button(separator, text="STOP", command=self.stop, width=10)
        stop_button.pack(side=tk.LEFT)
        
        clear_button = tk.Button(separator, text="CLEAR", command=self.clear, width=10)
        clear_button.pack(side=tk.LEFT)
        
        #self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'start processing\n')

        #self.onUpdate()
        
    def test(self):
        if not self.running:
            self.running=False
            self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'test processing\n')
            self.text.see(tk.END)
            self.onTest()
        
    def start(self):
        if not self.running:
            self.running=True
            self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'start processing\n')
            self.text.see(tk.END)
            # check and create result folder
            check_folder(folder=os.path.join(self.params.dirs['root'],self.params.dirs['classification']),create=True)
            if not check_folder(folder=os.path.join(self.params.dirs['root'],self.params.dirs['classification'],self.params.dirs['results']),create=True):
                self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'results folder is created\n')
                self.text.see(tk.END)
                print('result folders are checked and created')
            else:
                self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'results folder exists\n')
                self.text.see(tk.END)
            self.onUpdate()

    def stop(self):
        if self.running:
            self.running=False
            self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'stop processing\n')
            self.text.see(tk.END)
        
    def clear(self):
        self.text.delete('1.0', tk.END)
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d"))
     
    def onTest(self):
        test_dir=askdirectory()
        self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'processing: '+test_dir+'\n')
        self.text.see(tk.END)
#       measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        # get list of images to process
        df_images2process=images2process_list_in_depth(test_dir,file2check1=[],file2check2=['dummy'],level=1)
        if not df_images2process.empty:
            df_images_processed=self.process_measure(df_images2process)  
            print(df_images_processed)
            
        # write to csv
        df_images_processed.to_csv(os.path.join(test_dir,'classification_result.csv'))
            
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d")+' '+time.strftime("%I:%M:%S"))
             
    def onUpdate(self):
        measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        # get list of images to process
        df_images2process=images2process_list_in_depth(measure_dir,
                                                               file2check1=[self.params.files['control']],
                                                               file2check2=[self.params.files['measure']],
                                                               level=3)
        if not df_images2process.empty:
            df_images_processed=self.process_measure(df_images2process)  
            self.create_output(df_images_processed)
            
            # ToDo: add diagnostics: processing of N images took T secs.
            self.text.insert(tk.END, str(df_images_processed.shape[0])+' images were processed\n')
            self.text.see(tk.END)
        
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d")+' '+time.strftime("%I:%M:%S"))
             
        if self.running:
            self.after(1000,self.onUpdate)
    
    def process_oneimage(self,image_file):
        im = classifications.create_image(image_file,cropped=False)
        predicted_label, prob_trash = self.cnn_1.classify(im)
        if predicted_label==0:
            # TRASH goes to Recycle bin
            predicted_type=keysWithValue(self.params.type_dict_trash,str(predicted_label))[0]
            prob_taxon=0 # use np.NaN instead
            self.text.insert(tk.END, 'TRASH model result : '+'type : '+predicted_type+' ; prob : '+str(prob_trash)+'\n')
            self.text.see(tk.END)
        else:
            # NOT TRASH
            im = classifications.create_image(image_file,cropped=True)
            predicted_label, prob_taxon = self.cnn_2.classify(im)
            predicted_type=keysWithValue(self.params.type_dict_taxon,str(predicted_label))[0]
            self.text.insert(tk.END, 'TAXON model result : '+'type : '+predicted_type+' ; prob : '+str(prob_taxon)+'\n')
            self.text.see(tk.END)
            
        return predicted_type, prob_trash, prob_taxon
            
    def process_measure(self,df_images2process):
        df_images_processed=df_images2process.copy()
        df_images_processed['predicted_type']=None
        df_images_processed['prob_trash']=None
        df_images_processed['prob_taxon']=None
        for index, fo in df_images2process.iterrows():
           
            self.text.insert(tk.END, 'processing : '+fo['image_file']+'\n')
            
            image_fullfile=os.path.join(fo['root'],fo['image_file'])
            
            predicted_type, prob_trash, prob_taxon=self.process_oneimage(image_fullfile)
            
            df_images_processed['predicted_type'][index]=predicted_type
            df_images_processed['prob_trash'][index]=prob_trash
            df_images_processed['prob_taxon'][index]=prob_taxon
                        
        return df_images_processed
        
    def create_output(self,df_images_processed):

        measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        folders=df_images_processed['dir2'].unique()
        for fo in folders:
            df_temp=df_images_processed[df_images_processed['dir2']==fo]
            cur_dir=os.path.join(measure_dir,df_temp['dir1'][0],fo)
            self.text.insert(tk.END, 'visiting : '+cur_dir+'\n')

            res_folder1=os.path.join(self.params.dirs['root'],self.params.dirs['classification'],
                                self.params.dirs['results'],df_temp['dir1'][0])
            check_folder(folder=res_folder1,create=True)
            res_folder=os.path.join(res_folder1,fo)
            check_folder(folder=res_folder,create=True)

            for index, df_image in df_temp.iterrows():
                class_folder=os.path.join(res_folder,df_image['predicted_type'])
                check_folder(folder=class_folder,create=True)
                shutil.copy(os.path.join(df_image['root'],df_image['image_file']),
                            os.path.join(class_folder,df_image['image_file']))  
                
            file=open(os.path.join(cur_dir,'MeasureSum.xml'),'w')
            # ToDo: fill up MeasureSum with content
            file.close()
#            

if __name__ == "__main__":
    
    
#    logging.info('STARTING')

#   creating gui
    root = tk.Tk()
    app=Application(master=root)
    root.mainloop()

    
    
    
#    logging.info('FINISHING')
#    logging.shutdown()
#    os.remove(log_file)
#    sys.exit(1)


# ToDo: get date
# Check folders
# If there is a folder to process - do classification
# create file structure
# copy images to folders

# map file classified taxonname - taxonname

# classification class - input: image file name; output: class name (pct)
# stats class
# file manager class
