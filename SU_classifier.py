# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:59:11 2017

@author: SzMike
"""

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename      
import csv
import time
import os
import pandas as pd
import shutil
import xml.etree.ElementTree as ET
from src_tools.file_helper import read_log, images2process_list_in_depth, images2process_from_csv, check_folder
import classifications
from src_tools.results2xml import XMLWriter

import queue
import threading

#import sys
#import logging

# Logging setup
#log_file='progress.log'
#logging.basicConfig(filename=log_file,level=logging.DEBUG)

#def keysWithValue(aDict, target):
#    return sorted(key for key, value in aDict.items() if target == value)

class params:
    
    def __init__(self,cfg_file='cfg_SU_classifier.xml'):
        # reading parameters
        self.dirs={}
        self.files={}
        self.neural={}
        self.processing={}
        self.thresholds={}
        if os.path.exists(cfg_file):
            tree = ET.parse(cfg_file)
            self.dirs['root'] = tree.find('folders/root').text 
            self.dirs['measurement'] = tree.find('folders/measurement').text 
            self.dirs['classification'] = tree.find('folders/classification').text 
            self.dirs['results'] = tree.find('folders/results').text 
            self.files['control'] = tree.find('files/control').text 
            self.files['measure'] = tree.find('files/measure').text
            self.files['hologram'] = tree.find('files/hologram').text
            self.files['pred_strength_out'] = tree.find('files/pred_strength_out').text  
            self.files['model_taxon'] = os.path.join(tree.find('folders/model').text,
                                              tree.find('files/model_taxon').text)
            self.neural['model_output_layer'] = int(tree.find('neural/model_output_layer').text)
            self.files['typedict'] = os.path.join(tree.find('folders/model').text,
                                              tree.find('files/typedict').text)
            
            self.additional_classes=['Others.Others.Unsure','Others.Others.InappropriateSize']
            
            self.type_dict_taxon=self.get_typedict(self.files['typedict'])
                        
            self.files['threshold'] = os.path.join(tree.find('folders/model').text,
                                              tree.find('files/threshold').text)

            self.threshold_df_taxon=self.get_thresholds(self.files['threshold'])
   
            self.processing['auto_start'] = tree.find('processing/auto_start').text
            self.processing['ch_shift'] = tree.find('processing/ch_shift').text
            self.processing['save_cropped'] = tree.find('processing/save_cropped').text
            
#            print(self.dirs)
#            print(self.files)
#            print(self.neural)
#            print(self.processing)
#            print(self.thresholds)
     
#    def set_additional_classes(self,additional_classes):
#        self.additional_classes=additional_classes
        
    def get_classes(self):
        classes=list(self.type_dict_taxon.values())
        classes=classes+self.additional_classes
        return sorted(classes)
        
    def get_typedict(self,typedict_file):
        type_dict={}
        if os.path.isfile(typedict_file):
            with open(typedict_file, 'rt') as tf:
                reader =csv.DictReader(tf, delimiter=':')
                for row in reader:
                    type_dict[row['label']]=row['type']
            print('typeDict loaded')
        else:
            for i in range(100):
                type_dict[str(i)]=str(i)
        return type_dict
    
    def get_merge_dict(self,cur_dir):
        class_map_file=os.path.join(cur_dir,'class_map.csv')
        merge_dict={}
        try:
            with open(class_map_file, mode='r') as infile:
                reader = csv.reader(infile,delimiter=':')
                next(reader,None) # skip header
                for rows in reader:
                    if rows:
                        if rows[0][0]!='#':
                            merge_dict[rows[0]]=rows[1]
        except:
            print('missing class_map.csv')
        return merge_dict
    
    def get_thresholds(self,threshold_file):
        n_taxons=len(self.type_dict_taxon.values())
        ths_init=pd.DataFrame(data={'Taxon':list(self.type_dict_taxon.values()),
                                             'strength_tsh':[0]*n_taxons, # prediction strength
                                             'min_l':[0]*n_taxons, # min longer size
                                             'max_l':[10000]*n_taxons})
        ths_out=ths_init.copy()
        if os.path.isfile(threshold_file):
           ths=pd.read_csv(threshold_file,sep=',')
           print('thresholds loaded')
           for row_index, row in ths.iterrows():
               q=ths_init[ths_init['Taxon']==row['Taxon']]
               if not q.empty:
                   if 'max_l' in list(ths.columns.values):
                       ths_out.loc[q.index[0],'max_l']=row['max_l']
                   if 'min_l' in list(ths.columns.values):
                       ths_out.loc[q.index[0],'min_l']=row['min_l']
                   if 'strength_tsh' in list(ths.columns.values):
                       ths_out.loc[q.index[0],'strength_tsh']=row['strength_tsh']
#          
        return ths_out
                          
            
class process:
    def __init__(self,params):
        self.params=params
        self.cur_progress=0;
        self.elapsed=0;
        self.correct_RGBShift=self.params.processing['ch_shift']=='True'
        self.save_cropped=self.params.processing['save_cropped']=='True'
        self.cnn_taxon=classifications.cnn_classification(self.params.files['model_taxon'],model_output_layer=self.params.neural['model_output_layer'])
        self.classes=[]            
        print('load model '+self.params.files['model_taxon'])
        
    def process_oneimage(self,image_file):
        predicted_label=None
        predicted_type=None
        predicted_type=None
        final_type=None
        predicted_strength=None
        char_sizes=None
        
        if self.save_cropped:
            tmp_folder=os.path.join(os.path.curdir,'tmp_crop')
            check_folder(tmp_folder,create=True)
            save_file=os.path.join(tmp_folder,os.path.basename(image_file))
            category='cropped'
        else:
            save_file=''
            category=''
        img, char_sizes = classifications.create_image(image_file,cropped=True,correct_RGBShift=self.correct_RGBShift,
                                                       save_file=save_file,category=category)

        if img is not None and char_sizes is not None:
            predicted_label, predicted_strength = self.cnn_taxon.classify(img,char_sizes=char_sizes)
            # ToDo: implement threshold logic here on char_sizes and predicted strength - by class
            predicted_type=self.params.type_dict_taxon[str(predicted_label)]
            final_type=predicted_type
            row=self.params.threshold_df_taxon.loc[self.params.threshold_df_taxon['Taxon']==predicted_type]       
            min_l=row['min_l'].values[0]
            max_l=row['max_l'].values[0]
            strength_tsh=row['strength_tsh'].values[0]      
            
            if predicted_strength < strength_tsh: 
                final_type='Others.Others.Unsure'
            if char_sizes[0]<min_l: # char_sizes[0] larger axis length
                final_type='Others.Others.InappropriateSize'
            if char_sizes[0]>max_l:
                final_type='Others.Others.InappropriateSize'
                
                
#            self.text.insert(tk.END, 'TAXON model result : '+'type : '+predicted_type+' ; strength : '+str(strength_str)+'\n')
#            self.text.see(tk.END)
#        print(t-time.time()) 
        return predicted_type, final_type, predicted_strength, char_sizes
    
    def process_image_list(self,thread_queue=None,df_images2process=None):
    
        df_images_processed=df_images2process.copy()
        df_images_processed['predicted_type']=None
        df_images_processed['final_type']=None
        df_images_processed['predicted_strength']=None
        df_images_processed['char_size_long']=None
        df_images_processed['char_size_short']=None
        n_images2process=len(df_images2process)
        t=time.time()
        for index, fo in df_images2process.iterrows():
       
#        self.text.insert(tk.END, 'processing : '+fo['image_file']+'\n')
        
            self.cur_progress=float(index)/float(n_images2process)
            image_fullfile=os.path.join(fo['root'],fo['image_file'])
            
            predicted_type, final_type, predicted_strength, char_sizes=self.process_oneimage(image_fullfile)
            
            if predicted_type is not None and char_sizes is not None:
                df_images_processed['predicted_type'][index]=predicted_type
                df_images_processed['final_type'][index]=final_type
                df_images_processed['predicted_strength'][index]=predicted_strength
                df_images_processed['char_size_long'][index]=char_sizes[0]
                df_images_processed['char_size_short'][index]=char_sizes[1]
                        
        # put result to queue when finished
        self.elapsed=time.time() - t
        thread_queue.put(df_images_processed)

           

class Application(tk.Frame):
    #running=False
#    running_states=('checking','running','stopped','test')
    running_state='stopped'
#    proc_modes=('ws','test','evaluate')
    proc_mode='ws'
    cur_folder='.'
    test_dir=cur_folder
       
    def __init__(self, master=None):
        self.params=params()
        self.pred_strength_out=self.params.files['pred_strength_out']=='True'
        self.merge_dict={}
        if not self.params.dirs:
            print('config file - failed to load')
            return
        
        
        tk.Frame.__init__(self, master, background="yellow")
        self.pack(fill="both", expand=True)
        self.createWidgets()
        
        self.process=process(self.params)
        
        if self.params.processing['auto_start']=='True':
            self.start()
        else:
            self.running_state='stopped'
        
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
        self.date_entry.pack(fill=tk.X, side=tk.LEFT)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d"))
        
        self.state_entry = tk.Entry(self, bg='yellow')
        self.state_entry.pack(fill=tk.Y, side=tk.RIGHT)
        self.state_entry.insert(0, self.running_state)
        
        self.progress = ttk.Progressbar(self, orient='horizontal',
                                        length=200, mode='determinate')
        self.progress.pack(fill=tk.Y, side=tk.RIGHT)
        self.progress['maximum'] = 1
        
        test_button = tk.Button(separator, text="TEST", command=self.test, width=10)
        test_button.pack(side=tk.RIGHT)
        
        start_button = tk.Button(separator, text="EVALUATE", command=self.evaluate, width=10)
        start_button.pack(side=tk.RIGHT)
        
        start_button = tk.Button(separator, text="MEASURE", command=self.start, width=10)
        start_button.pack(side=tk.LEFT)
        
        stop_button = tk.Button(separator, text="STOP", command=self.stop, width=10)
        stop_button.pack(side=tk.LEFT)
        
        clear_button = tk.Button(separator, text="CLEAR", command=self.clear, width=10)
        clear_button.pack(side=tk.LEFT)
        
        #self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'start processing\n')

        #self.onUpdate()
        
    def change_state(self,to_state):
        # ToDo: check if state exists
            self.running_state=to_state;
            self.state_entry.delete(0, tk.END)
            self.state_entry.insert(0, self.running_state)        
        
    def test(self):
        if self.running_state=='stopped':
            self.change_state('checking')
            self.proc_mode='test'
            self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'test processing\n')
            self.text.see(tk.END)
            self.onTest()
            self.change_state('stopped')
            
    def evaluate(self):
        if self.running_state=='stopped':
            self.change_state('checking')
            self.proc_mode='evaluate'
            self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'evaluating\n')
            self.text.see(tk.END)
            self.onEvaluate()
            self.change_state('stopped')
        
    def start(self):
        if self.running_state=='stopped':
            self.change_state('checking')
            self.proc_mode='ws'
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
        if self.running_state=='checking':
            self.change_state('stopped')
            self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'stop processing\n')
            self.text.see(tk.END)
        
    def clear(self):
        self.text.delete('1.0', tk.END)
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d"))
     
    def onTest(self):
        self.test_dir=askdirectory(title = 'Open a test directory')
        self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'processing: '+self.test_dir+'\n')
        self.text.see(tk.END)
#       measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        # get list of images to process
        if os.path.exists(self.test_dir):
            df_images2process=images2process_list_in_depth(self.test_dir,file2check1=[],file2check2=['dummy'],level=1)
            if not df_images2process.empty:
                self.text.insert(tk.END, str(len(df_images2process))+' images are being processed\n')
            
                self.thread_queue = queue.Queue()
                self.thread = threading.Thread(target=self.process.process_image_list, args=(self.thread_queue, df_images2process))
                self.change_state('running')

                self.thread.start()
                self.after(100, self.listen_for_result)
                
                # write to csv
                
                    
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d")+' '+time.strftime("%I:%M:%S"))
        
    def onEvaluate(self):
        self.eval_dir=askdirectory(title = 'Open a directory to evaluate')
        self.eval_file=askopenfilename(title='Choose an image list to process',filetypes=[('.csv',('*.csv',))])
        self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'processing: '+self.eval_file+'\n')
        self.text.see(tk.END)
#       measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        # get list of images to process
        if os.path.exists(self.eval_dir) and os.path.exists(self.eval_file):
            self.merge_dict=self.params.get_merge_dict(self.eval_dir)
            df_images2process=images2process_from_csv(self.merge_dict,self.eval_file,self.eval_dir)
            if not df_images2process.empty:
                self.text.insert(tk.END, str(len(df_images2process))+' images are being processed\n')
            
                self.thread_queue = queue.Queue()
                self.thread = threading.Thread(target=self.process.process_image_list, args=(self.thread_queue, df_images2process))
                self.change_state('running')

                self.thread.start()
                self.after(100, self.listen_for_result)
                
                # write to csv
                
                    
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d")+' '+time.strftime("%I:%M:%S"))
             
    def onUpdate(self):
        measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        # get list of images to process
        file2check1=[]
        if self.params.files['control']:
            file2check1=[self.params.files['control']]
        df_images2process=images2process_list_in_depth(measure_dir,
                                                               file2check1=file2check1,
                                                               file2check2=[self.params.files['measure']],
                                                               file2exclude=[self.params.files['hologram']],
                                                               level=3)
        if not df_images2process.empty:
#            t=time.time()
            self.text.insert(tk.END, str(len(df_images2process))+' images are being processed\n')
            
            self.thread_queue = queue.Queue()
            self.thread = threading.Thread(target=self.process.process_image_list, args=(self.thread_queue, df_images2process))
            self.change_state('running')

            self.thread.start()
            self.after(100, self.listen_for_result)
        
        self.date_entry.delete(0, tk.END)
        self.date_entry.insert(0, time.strftime("%Y/%m/%d")+' '+time.strftime("%H:%M:%S"))
             
        if self.running_state=='checking':
            self.after(1000,self.onUpdate)
    
    def listen_for_result(self):
        '''
        Check if there is something in the queue
        '''
        try:
            df_images_processed = self.thread_queue.get(0)
            if self.proc_mode=='test':
                df_images_processed.to_csv(os.path.join(self.test_dir,'classification_result.csv'),index=False)
            elif self.proc_mode=='evaluate':
                df_images_processed.to_csv(os.path.join(self.eval_dir,'classification_result.csv'),index=False)
                cont_table=self.create_contingency(df_images_processed)
                self.text.insert(tk.END, self.date_entry.get()+' '+time.strftime("%I:%M:%S")+' : '+'creating contingency table..\n')
                self.text.see(tk.END)

                cont_table.to_csv(os.path.join(self.eval_dir,'contingency_table.csv'),index=True)
        
            else:
                res_folder=os.path.join(self.params.dirs['root'],
                                         self.params.dirs['classification'],
                                         self.params.dirs['results'])
                check_folder(folder=res_folder,create=True)
                df_images_processed.to_csv(os.path.join(res_folder,'classification_result.csv'),index=False)
                self.create_output(df_images_processed)
            self.thread_queue.task_done()
            self.change_state('checking')
            
#            elapsed = time.time() - t
            self.progress['value']=1
            self.text.insert(tk.END, str(df_images_processed.shape[0])+' images were processed in '+
                                         str("{0:.2f}".format(self.process.elapsed))+' secs\n')
            
            self.text.see(tk.END)
            if self.proc_mode=='test' or self.proc_mode=='evaluate':
                self.stop()
            else:
                self.onUpdate()
            
        except queue.Empty:
            self.change_state('running')
            self.progress['value']=self.process.cur_progress
            self.after(1000, self.listen_for_result)
    
    def create_contingency(self,df_images_processed):
        sorted_classes=self.params.get_classes()
        #num_classes=len(sorted_classes)
        cont_table=pd.DataFrame(0,index=sorted_classes,    
              columns=sorted_classes)


        for index, row in df_images_processed.iterrows():

            cont_table[row['dir2']][row['final_type']]+=1
            # rows are actual labels, cols are predictions,     

        return cont_table
        
    def create_output(self,df_images_processed):

        measure_dir=os.path.join(self.params.dirs['root'],self.params.dirs['measurement'])
        date_folders=df_images_processed['dir1'].unique()
        
        for dfo in date_folders:
            df_date=df_images_processed[df_images_processed['dir1']==dfo]
            measure_folders=df_date['dir2'].unique()
            for mfo in measure_folders:
                # select items
                df_measure=df_date[df_date['dir2']==mfo]
 
                # Check and create folders
                cur_dir=os.path.join(measure_dir,dfo,mfo)
                self.text.insert(tk.END, 'visiting : '+cur_dir+'\n')
    
                res_folder1=os.path.join(self.params.dirs['root'],
                                         self.params.dirs['classification'],
                                         self.params.dirs['results'],dfo)
                check_folder(folder=res_folder1,create=True)
                res_folder=os.path.join(res_folder1,mfo)
                check_folder(folder=res_folder,create=True)
              
                # Write to folders
    
                for index, df_image in df_measure.iterrows():
                    class_folder=os.path.join(res_folder,df_image['final_type'])
                    check_folder(folder=class_folder,create=True)
                    if self.pred_strength_out:
                        strength_str='{0:0{1}d}'.format(int(round(df_image['predicted_strength'])),5)+'_'
                    else:
                        strength_str=''
                    
                    shutil.copy(os.path.join(df_image['root'],df_image['image_file']),
                                os.path.join(class_folder,strength_str+df_image['image_file']))  
                
                # Read log
                if self.params.files['control']:
                    log_file=os.path.join(measure_dir,dfo,mfo,self.params.files['control'])
                    log_dict=read_log(log_file)
                    
                    measured_volume=float(log_dict['Measured Volume'].split(' ')[0])
                    measure_datetime=log_dict['Sample DateTime'].split('\t')
                    measure_date=measure_datetime[0].replace('.','')
                    measure_time=measure_datetime[1].replace(':','')
    
                    if not (type(measure_time) == str):
                        measure_time=time.strftime("%H%M")
                        
                    if not (type(measure_date) == str):
                        measure_date=time.strftime("%Y%m%d_%H%M")
                    
                    scaled=True
                    if not (measured_volume>0):
                        scaled=False
                    
                    # Create XML            
                    cur_result = XMLWriter()
                    cur_result.addAllCount(df_measure.shape[0],False)
                    if scaled:
                        cur_result.addAllCount(int(float(df_measure.shape[0])/measured_volume),True)
                        cur_result.addMeasuredVolume(measured_volume)
      
                    taxons=df_measure.predicted_type.unique()
                    for taxon in taxons:
                        count=df_measure.predicted_type.value_counts()[taxon]
                        cur_result.addTaxonStat(taxon,int(count),False)
                        if scaled:
                            cur_result.addTaxonStat(taxon,int(float(count)/measured_volume),True)
      
                    
                    xml_file=os.path.join(measure_dir,dfo,mfo,'MeasureSum_'+measure_date+'_'+measure_time+'.xml')
                    cur_result.save(targetFile=xml_file)
                    
                    xml_file_details=os.path.join(measure_dir,dfo,mfo,'MeasureDetails_'+measure_date+'_'+measure_time+'.xml')
                    file=open(xml_file_details,'w')
                    file.close()
                else:
                    xml_file=os.path.join(measure_dir,dfo,mfo,'MeasureSum_Zero.xml')
                    file=open(xml_file,'w')
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
