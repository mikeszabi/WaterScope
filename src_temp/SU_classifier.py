# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:59:11 2017

@author: SzMike
"""

import tkinter as tk
import time
import os
import shutil
import csv
import xml.etree.ElementTree as ET
from file_helper import images2process_list_in_depth, check_folder
from classifications import create_image, cnn_classification

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
        tree = ET.parse('cfg.xml')
        self.dirs={}
        self.files={}
        self.dirs['root'] = tree.find('folders/root').text 
        self.dirs['measurement'] = tree.find('folders/measurement').text 
        self.dirs['classification'] = tree.find('folders/classification').text 
        self.dirs['results'] = tree.find('folders/results').text 
        self.files['control'] = tree.find('files/control').text 
        self.files['measure'] = tree.find('files/measure').text
        self.files['model'] = os.path.join(tree.find('folders/model').text,
                                          tree.find('files/model').text)
        self.files['typedict'] = os.path.join(tree.find('folders/model').text,
                                          tree.find('files/typedict').text)
        
        self.type_dict=self.get_typedict(self.files['typedict'])
        
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
    running=True
    cur_folder='.'
       
    def __init__(self, master=None):
        self.params=params()
        
        self.cnn=cnn_classification(self.params.files['model'])
        print('load model '+self.params.files['model'])
        
        # check result folder
        check_folder(folder=os.path.join(self.params.dirs['root'],self.params.dirs['classification']),create=True)
        check_folder(folder=os.path.join(self.params.dirs['root'],self.params.dirs['classification'],self.params.dirs['results']),create=True)
        print('result folders are checked and created')
        
        tk.Frame.__init__(self, master, background="green")
        self.pack(fill="both", expand=True)
        self.createWidgets()

     
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
        
        start_button = tk.Button(separator, text="START", command=self.start, width=10)
        start_button.pack(side=tk.LEFT)
        
        stop_button = tk.Button(separator, text="STOP", command=self.stop, width=10)
        stop_button.pack(side=tk.LEFT)
        
        clear_button = tk.Button(separator, text="CLEAR", command=self.clear, width=10)
        clear_button.pack(side=tk.LEFT)
        
        self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'start processing\n')

        self.onUpdate()
        
    def start(self):
        if not self.running:
            self.running=True
            self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'start processing\n')
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
        
    def onUpdate(self):
        measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        # get list of images to process
        images2process_list_indir=images2process_list_in_depth(measure_dir,
                                                               file2check1=[self.params.files['control']],
                                                               file2check2=[self.params.files['measure']],
                                                               level=3)
        if images2process_list_indir:
            self.process(images2process_list_indir)      
            
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d")+' '+time.strftime("%I:%M:%S"))
             
        if self.running:
            self.after(1000,self.onUpdate)
            
    def process(self,images2process_list_indir):
        measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        for fo in images2process_list_indir:
            cur_dir=os.path.join(measure_dir,fo[0],fo[1])
            self.text.insert(tk.END, 'visiting : '+cur_dir+'\n')
            res_folder1=os.path.join(self.params.dirs['root'],self.params.dirs['classification'],
                                self.params.dirs['results'],fo[0])
            check_folder(folder=res_folder1,create=True)
            res_folder=os.path.join(res_folder1,fo[1])
            check_folder(folder=res_folder,create=True)

            for image in fo[2]:
                self.text.insert(tk.END, 'processing : '+image+'\n')
                
                image_file=os.path.join(measure_dir,fo[0],fo[1],image)
                im = create_image(image_file)
                predicted_label, prob = self.cnn.classify(im)
                predicted_type=keysWithValue(self.params.type_dict,str(predicted_label))[0]
                self.text.insert(tk.END, 'result : '+'type : '+predicted_type+' ; prob : '+str(prob)+'\n')
                self.text.see(tk.END)
                
                class_folder=os.path.join(res_folder,predicted_type)
                check_folder(folder=class_folder,create=True)
                shutil.copy(image_file,os.path.join(class_folder,image))               
  
            file=open(os.path.join(cur_dir,'MeasureSum.xml'),'w')
            file.close()
            

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
