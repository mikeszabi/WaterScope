# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:42:51 2016

@author: picturio
"""

    
import os
import numpy as np
import matplotlib.pyplot as plt

#from cntk.device import try_set_default_device, gpu
from cntk import cross_entropy_with_softmax, classification_error, input_variable, softmax, element_times, reduce_mean
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs, transforms
from cntk import Trainer
from cntk import momentum_sgd, learning_rate_schedule, learning_parameter_schedule, learners.momentum_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter, TensorBoardProgressWriter


from src_train.model_functions import create_basic_model, create_advanced_model
from src_train.train_config import train_params

data_dir=os.path.join(r'C:\Users','picturio','OneDrive\WaterScope')
cfg=train_params(data_dir,crop=True,training_id='20171113')

model_file=os.path.join(cfg.train_dir,'cnn_model.dnn')
model_temp_file=os.path.join(cfg.train_dir,'cnn_model_temp.dnn')
train_log_file=os.path.join(cfg.train_log_dir,'progress_log.txt')

train_map=os.path.join(cfg.train_dir,'train_map.txt')
test_map=os.path.join(cfg.train_dir,'test_map.txt')
# GET train and test map from prepare4train

data_mean_file=os.path.join(cfg.train_dir,'data_mean.xml')

# model dimensions

image_height = 64
image_width  = 64
num_channels = 3
num_classes  = 32



#
# Define the reader for both training and evaluation action.
#
def create_reader(map_file, mean_file, train, image_height=64, image_width=64, num_channels=3, num_classes=32):
  
    # transformation pipeline for the features has jitter/crop only when training
    # https://docs.microsoft.com/en-us/python/api/cntk.io.transforms?view=cntk-py-2.2
    trs = []
    if train:
        trs += [
            transforms.crop(crop_type='randomside', side_ratio=0, jitter_type='none') # Horizontal flip enabled
        ]
    trs += [
        transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        transforms.mean(mean_file)
    ]
    # deserializer
    image_source=ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=trs), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    ))
    return MinibatchSource(image_source)
    
#
# Train and evaluate the network.
#

    
reader_train = create_reader(train_map, data_mean_file, True, image_height=image_height, image_width=image_width, num_channels=num_channels, num_classes=num_classes)
reader_test  = create_reader(test_map, data_mean_file, False,  image_height=image_height, image_width=image_width, num_channels=num_channels, num_classes=num_classes)

#==============================================================================
# SET parameters
#==============================================================================
max_epochs=300
model_func=create_basic_model


#==============================================================================
# ###
#==============================================================================
input_var = input_variable((num_channels, image_height, image_width))
label_var = input_variable((num_classes))

# Normalize the input
feature_scale = 1.0 / 256.0
input_var_norm = element_times(feature_scale, input_var)

# apply model to input
z = model_func(input_var_norm, out_dims=num_classes)

"""
# Training action
"""

# loss and metric
ce = cross_entropy_with_softmax(z, label_var)
pe = classification_error(z, label_var)
#pe5 = classification_error(z, label_var, topN=5)

# training config
epoch_size     = 48000
minibatch_size = 128

# Set training parameters
lr_per_mb              = [0.01]*25 + [0.001]*25 + [0.0001]*25 + [0.00001]*25 + [0.000001]
lr_schedule            = learning_parameter_schedule(lr_per_mb, minibatch_size=minibatch_size, epoch_size=epoch_size)
mm_schedule            = learners.momentum_schedule(0.9, minibatch_size=minibatch_size)
#momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
l2_reg_weight          = 0.0005

# trainer objectS
progress_writers = [ProgressPrinter(tag='Training', num_epochs=max_epochs)]
#progress_writers = [ProgressPrinter(tag='Training', log_to_file=train_log_file, num_epochs=max_epochs, gen_heartbeat=True)]

if cfg.train_log_dir is not None:
    tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=cfg.train_log_dir, model=z)
    progress_writers.append(tensorboard_writer)
    
#learner     = momentum_sgd(z.parameters, 
#                           lr = lr_per_mb, momentum = momentum_time_constant, 
#                           l2_regularization_weight=l2_reg_weight)

learner = momentum_sgd(z.parameters, 
                       lr_schedule, mm_schedule, 
                       minibatch_size=minibatch_size, 
                       unit_gain=False, 
                       l2_regularization_weight=l2_reg_weight)


######### RESTORE TRAINER IF NEEDED
trainer     = Trainer(z, (ce, pe), learner, progress_writers)
# trainer.restore_from_checkpoint(model_temp_file)

# define mapping from reader streams to network inputs
input_map = {
    input_var: reader_train.streams.features,
    label_var: reader_train.streams.labels
}

log_number_of_parameters(z) ; print()

# perform model training
batch_index = 0
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
    trainer.save_checkpoint(model_temp_file)
    
#
# Evaluation action
#
epoch_size     = 16000
minibatch_size = 64

# process minibatches and evaluate the model
metric_numer    = 0
metric_denom    = 0
sample_count    = 0
minibatch_index = 0

input_map = {
    input_var: reader_test.streams.features,
    label_var: reader_test.streams.labels
}

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

print("")
print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
print("")

# Visualize training result:
window_width            = 32
loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0)) 
error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0)) 

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

pred=softmax(z)

pred.save(model_file)