# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:59:11 2017

@author: SzMike
"""

import tkinter as tk
import time
import os
import xml.etree.ElementTree as ET
from file_helper import *

#import sys
#import logging

# Logging setup
#log_file='progress.log'
#logging.basicConfig(filename=log_file,level=logging.DEBUG)

class params:
    
    def __init__(self):
        # reading parameters
        tree = ET.parse('cfg.xml')
        self.dirs={}
        self.files={}
        self.dirs['root'] = tree.find('folders/root').text 
        self.dirs['measurement'] = tree.find('folders/measurement').text 
        self.dirs['classification'] = tree.find('folders/classification').text 
        self.files['control'] = tree.find('files/control').text 
                

class Application(tk.Frame):
    running=True
    cur_folder='.'
       
    def __init__(self, master=None):
        self.params=params()
        
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
        
        self.onUpdate()
        
    def start(self):
        self.running=True
        self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'start processing\n')
        self.text.see(tk.END)
        self.onUpdate()

    def stop(self):
        self.running=False
        self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'stop processing\n')
        self.text.see(tk.END)
        
    def clear(self):
        self.text.delete('1.0', tk.END)
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d"))
        
    def onUpdate(self):
        # ToDo: update date entry with clear on new days
        date_str=self.date_entry.get().replace('/','')
        self.cur_process_folder=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'],date_str)
        if check_folder(folder=self.cur_process_folder,create=False):
            self.text.insert(tk.END, 'searching in : '+self.cur_process_folder+'\n')
            self.text.see(tk.END)
        else:
            self.text.insert(tk.END, 'non existent : '+self.cur_process_folder+'\n')
            self.text.see(tk.END)
             
        if self.running:
            self.after(1000,self.onUpdate)

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
