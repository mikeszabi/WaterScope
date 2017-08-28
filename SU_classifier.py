# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:59:11 2017

@author: SzMike
"""

import tkinter as tk

import sys
import os
import logging


import xml.etree.ElementTree as ET

# Logging setup
#log_file='progress.log'
#logging.basicConfig(filename=log_file,level=logging.DEBUG)

class SU_gui(tk.Frame):
    running=True
    #text=None
       
    def __init__(self, master):
        tk.Frame.__init__(self, master, background="green")
        
        scrollbar = tk.Scrollbar(self)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text=tk.Text(self,height=10,width=50, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.text.pack(expand=True, fill='both')
        scrollbar.config(command=self.text.yview)


        separator = tk.Frame(height=2, bd=1, relief=tk.SUNKEN)
        separator.pack(fill=tk.X, padx=5, pady=5)
        start_button = tk.Button(separator, text="START", command=self.start, width=10)
        #start_button.grid(row=0, column=0, sticky=tk.W)
        start_button.pack(side=tk.LEFT)
        stop_button = tk.Button(separator, text="STOP", command=self.stop, width=10)
        stop_button.pack(side=tk.LEFT)
        #stop_button.grid(row=1, column=0, sticky=tk.W)
        
        
        

        
    def start(self):
        self.running=True
        self.text.insert(tk.END, 'kulipintyó\n')
        self.text.see(tk.END)

    def stop(self):
        self.running=False
        self.text.delete('1.0', tk.END)
        


if __name__ == "__main__":
    
    
#    logging.info('STARTING')

#   creating gui
    root = tk.Tk()
    su=SU_gui(root)
    su.pack(fill="both", expand=True)
    root.mainloop()
    
    # reading parameters
    tree = ET.parse('cfg.xml')
    textelem = tree.find('folders/root') # findall if many
    textelem.text
    textelem = tree.find('processing/delete')
    
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
