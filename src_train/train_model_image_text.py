# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:42:51 2016

@author: picturio
"""
#training_id='20171126-Gray'
#num_classes  = 30
    
import os
import numpy as np
import matplotlib.pyplot as plt

#from cntk.device import try_set_default_device, gpu
from cntk import cross_entropy_with_softmax, classification_error, input_variable, softmax, element_times
from cntk import Trainer, UnitType
from cntk import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
#from cntk.learners import momentum_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter, TensorBoardProgressWriter
from cntk.ops import minus

from src_train.model_functions import create_model_ext
from src_train.train_config import train_params
from src_train.readers import create_reader_with_ext_values
#from src_train.readers import create_reader


#==============================================================================
# SET THESE PARAMETERS!
#==============================================================================
data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
# model dimensions

image_height = 64
image_width  = 64
num_channels = 3
numFeature = image_height * image_width * num_channels

#==============================================================================
# SET training parameters
#==============================================================================
max_epochs=150
model_func=create_model_ext

epoch_size     = 48000 # training
minibatch_size = 128 # training

#==============================================================================
# RUN CONFIG
#==============================================================================


cfg=train_params(data_dir,training_id=training_id)

train_map_o=os.path.join(cfg.train_dir,'train_map.txt')
test_map_o=os.path.join(cfg.train_dir,'test_map.txt')
#train_regr_labels=os.path.join(train_dir,'train_regrLabels.txt')
data_mean_file=os.path.join(cfg.train_dir,'data_mean.xml')
model_file=os.path.join(cfg.train_dir,'cnn_model.dnn')
model_temp_file=os.path.join(cfg.train_dir,'cnn_model_temp.dnn')
train_log_file=os.path.join(cfg.train_log_dir,'progress_log.txt')

train_map_image=os.path.join(cfg.train_dir,'train_map_image.txt')
test_map_image=os.path.join(cfg.train_dir,'test_map_image.txt')
train_map_text=os.path.join(cfg.train_dir,'train_map_text.txt')
test_map_text=os.path.join(cfg.train_dir,'test_map_text.txt')


#
# Evaluation action
#
def evaluate_test(input_map,reader_test,trainer,plot_data,epoch_size = 16000, minibatch_size=64,visualize=False):

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0
    
    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)
    
        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)
    
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
    
        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1
    
    err=(metric_numer*100.0)/metric_denom
    print("")
    print("Evaluation Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, err, metric_denom))
    print("")
    
    # Visualize training result:
    window_width            = 32
    loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0)) 
    error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0)) 
    
    if visualize:
        # Moving average.
        plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
        plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
        plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Loss')
        plt.title('Minibatch run vs. Training loss ')
        
        plt.show()
        
        plt.subplot(212)
        plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Label Prediction Error')
        plt.title('Minibatch run vs. Label Prediction Error ')
        plt.show()
    
    return err
    
#
# Train and evaluate the network.
#

    
reader_train = create_reader_with_ext_values(train_map_image, train_map_text, data_mean_file, True, image_height=image_height, image_width=image_width, num_channels=num_channels, num_classes=num_classes)
reader_test  = create_reader_with_ext_values(test_map_image, test_map_text,data_mean_file, False,  image_height=image_height, image_width=image_width, num_channels=num_channels, num_classes=num_classes)



#==============================================================================
# ###
#==============================================================================
# Normalize the inputs
feature_scale = 1.0 / 256.0
l_scale = 1.0 / 1000.0


input_var = input_variable((num_channels, image_height, image_width))
input_var_mean = minus(input_var,128)
input_var_norm = element_times(feature_scale, input_var_mean)

label_var = input_variable((num_classes))
size_var = input_variable((2))
size_var_norm =  element_times(l_scale, size_var)

# apply model to input
z = model_func(input_var_norm, size_var_norm, out_dims=num_classes)

"""
Training action
"""

# loss and metric
ce = cross_entropy_with_softmax(z, label_var)
pe = classification_error(z, label_var)
#pe5 = classification_error(z, label_var, topN=5)



# Set training parameters
#lr_per_minibatch              = [0.01]*25 + [0.001]*25 + [0.0001]*25 + [0.00001]*25 + [0.000001]
lr_per_minibatch       =  learning_rate_schedule([0.01]*25 + [0.003]*25 + [0.001]*25 + [0.0003],  UnitType.minibatch, epoch_size)
#lr_schedule            = learning_parameter_schedule(lr_per_mb, minibatch_size=minibatch_size, epoch_size=epoch_size)
#mm_schedule            = momentum_schedule(0.9, minibatch_size=minibatch_size)
momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
l2_reg_weight          = 0.001

# trainer objectS
progress_writers = [ProgressPrinter(tag='Training', num_epochs=max_epochs)]
#progress_writers = [ProgressPrinter(tag='Training', log_to_file=train_log_file, num_epochs=max_epochs, gen_heartbeat=True)]

if cfg.train_log_dir is not None:
    tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=cfg.train_log_dir, model=z)
    progress_writers.append(tensorboard_writer)


learner     = momentum_sgd(z.parameters, 
                           lr = lr_per_minibatch, momentum = momentum_time_constant, 
                           l2_regularization_weight=l2_reg_weight)



######### RESTORE TRAINER IF NEEDED
trainer     = Trainer(z, (ce, pe), learner, progress_writers)
# trainer.restore_from_checkpoint(model_temp_file)

# define mapping from reader streams to network inputs
input_map = {
    input_var: reader_train.streams.features,
    label_var: reader_train.streams.labels,
    size_var: reader_train.streams.ext_values,
}

log_number_of_parameters(z) ; print()

# perform model training
batch_index = 0
eval_errors=[]
plot_data = {'batchindex':[], 'loss':[], 'error':[]}
for epoch in range(max_epochs):       # loop over epochs
    sample_count = 0
    ev_avg=0
    i_count=0
    while sample_count < epoch_size:  # loop over minibatches in the epoch
        data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
        trainer.train_minibatch(data)                                   # update model with it

        sample_count += data[label_var].num_samples                     # count samples processed so far
        
        # For visualization...            
        plot_data['batchindex'].append(batch_index)
        plot_data['loss'].append(trainer.previous_minibatch_loss_average)
        plot_data['error'].append(trainer.previous_minibatch_evaluation_average)
        
        progress_writers[0].update_with_trainer(trainer, with_metric=True) # log progress
        batch_index += 1
        ev_avg+=trainer.previous_minibatch_evaluation_average
        i_count+=1
#    if ev_avg/i_count < 0.02:
#        break
    progress_writers[0].epoch_summary(with_metric=True)
#    trainer.summarize_training_progress()
#    if tensorboard_writer:
#        for parameter in z.parameters:
#            tensorboard_writer.write_value(parameter.uid + "/mean", reduce_mean(parameter).eval(), epoch)
#    print(epoch,' : ',ev_avg/i_count)
    if epoch % 10 ==0:
        trainer.save_checkpoint(model_temp_file)
        eval_errors.append(evaluate_test(input_map,reader_test,trainer,plot_data,
                          epoch_size=int(((1-cfg.trainRatio)/cfg.trainRatio)*epoch_size)))
        if len(eval_errors)>1:
            if eval_errors[-2]<eval_errors[-1]:
                print('reached max learning')
                break

evaluate_test(input_map,reader_test,trainer,plot_data,
                          epoch_size=int(((1-cfg.trainRatio)/cfg.trainRatio)*epoch_size),visualize=True)

pred=softmax(z)

pred.save(model_file)