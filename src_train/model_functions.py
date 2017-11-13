# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:44:25 2017

@author: picturio
"""
from cntk import relu
from cntk.layers import Convolution, MaxPooling, Dropout, Dense, For, Sequential, default_options
from cntk.initializer import glorot_uniform

def create_shallow_model(input, out_dims):
    
    convolutional_layer_1  = Convolution((5,5), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(input)
    pooling_layer_1  = MaxPooling((2,2), strides=(2,2))(convolutional_layer_1 )
    
    convolutional_layer_2 = Convolution((9,9), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_1)
    pooling_layer_2 = MaxPooling((2,2), strides=(2,2))(convolutional_layer_2)
 
    fully_connected_layer_1  = Dense(256, init=glorot_uniform())(pooling_layer_2)   
    fully_connected_layer_2  = Dense(128, init=glorot_uniform())(fully_connected_layer_1)
    dropout_layer = Dropout(0.5)(fully_connected_layer_2)

    output_layer = Dense(out_dims, init=glorot_uniform(), activation=None)(dropout_layer)
    
    return output_layer


def create_basic_model(input, out_dims):
    
    convolutional_layer_1  = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(input)
    pooling_layer_1  = MaxPooling((2,2), strides=(2,2))(convolutional_layer_1 )

    convolutional_layer_2 = Convolution((7,7), 64, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_1)
    pooling_layer_2 = MaxPooling((2,2), strides=(2,2))(convolutional_layer_2)

    convolutional_layer_2 = Convolution((9,9), 64, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_1)
    pooling_layer_2 = MaxPooling((2,2), strides=(2,2))(convolutional_layer_2)

    convolutional_layer_3 = Convolution((11,11), 64, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_2)
    pooling_layer_3 = MaxPooling((3,3), strides=(2,2))(convolutional_layer_3)
##    
    fully_connected_layer  = Dense(1024, init=glorot_uniform())(pooling_layer_3)
    dropout_layer_1 = Dropout(0.5)(fully_connected_layer)
    
    fully_connected_layer_2  = Dense(512, init=glorot_uniform())(dropout_layer_1)

    dropout_layer_2 = Dropout(0.5)(fully_connected_layer_2)

    output_layer = Dense(out_dims, init=glorot_uniform(), activation=None)(dropout_layer_2)
    
    return output_layer

def create_advanced_model(input, out_dims):
    
    with default_options(activation=relu):
        model = Sequential([
            For(range(2), lambda i: [  # lambda with one parameter
                Convolution((3,3), [32,64][i], pad=True),  # depth depends on i
                Convolution((7,7), [32,64][i], pad=True),
                Convolution((9,9), [32,64][i], pad=True),            
                MaxPooling((3,3), strides=(2,2))
            ]),
            For(range(2), lambda : [   # lambda without parameter
                Dense(512),
                Dropout(0.5)
            ]),
            Dense(out_dims, activation=None)
        ])
    output_layer=model(input)
    
    return output_layer