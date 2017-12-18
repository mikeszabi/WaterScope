# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:44:25 2017

@author: picturio
"""
from cntk import relu, splice
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
    
    convolutional_layer_1  = Convolution((7,7), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(input)

    pooling_layer_1  = MaxPooling((2,2), strides=(2,2))(convolutional_layer_1)

    convolutional_layer_2 = Convolution((5,5), 64, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_1)
    pooling_layer_2 = MaxPooling((2,2), strides=(2,2))(convolutional_layer_2)

    convolutional_layer_3 = Convolution((3,3), 96, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_2)

    pooling_layer_3 = MaxPooling((2,2), strides=(1,1))(convolutional_layer_3)

##    
    fully_connected_layer_1  = Dense(512, init=glorot_uniform())(pooling_layer_3)
    
    fully_connected_layer_2  = Dense(256, init=glorot_uniform())(fully_connected_layer_1)

    dropout_layer_1 = Dropout(0.5)(fully_connected_layer_2)

    output_layer = Dense(out_dims, init=glorot_uniform(), activation=None)(dropout_layer_1)
    
    return output_layer

def create_advanced_model(input, out_dims):
    
    with default_options(activation=relu):
        model = Sequential([
            For(range(2), lambda i: [  # lambda with one parameter
                Convolution((3,3), [16,32][i], pad=True),  # depth depends on i
                Convolution((5,5), [32,64][i], pad=True),
                Convolution((7,7), [32,64][i], pad=True),            
                MaxPooling((3,3), strides=(2,2))
            ]),
            For(range(2), lambda i: [   # lambda without parameter
                Dense([2048,512][i]),
                Dropout(0.5)
            ]),
            Dense(out_dims, activation=None)
        ])
    output_layer=model(input)
    
    return output_layer

def create_model_ext(input, ext_values, out_dims):
    
    # in VGG style
    #https://www.cs.toronto.edu/~frossard/post/vgg16/
    convolutional_layer_1_1  = Convolution((3,3), 16, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(input)
    convolutional_layer_1_2 = Convolution((5,5), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(convolutional_layer_1_1)
    pooling_layer_1 = MaxPooling((2,2), strides=(2,2))(convolutional_layer_1_2)

    convolutional_layer_2_1 = Convolution((3,3), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_1)
    convolutional_layer_2_2 = Convolution((7,7), 64, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(convolutional_layer_2_1)
    pooling_layer_2 = MaxPooling((2,2), strides=(1,1))(convolutional_layer_2_2)

    convolutional_layer_3_1 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_2)
    convolutional_layer_3_2 = Convolution((7,7), 96, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(convolutional_layer_3_1)
    pooling_layer_3 = MaxPooling((2,2), strides=(1,1))(convolutional_layer_3_2)

    convolutional_layer_4_1 = Convolution((3,3), 96, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_3)
    pooling_layer_4 = MaxPooling((2,2), strides=(1,1))(convolutional_layer_4_1)

##    
    fully_connected_layer_1  = Dense(512, init=glorot_uniform())(pooling_layer_4)
    dropout_layer_1 = Dropout(0.5)(fully_connected_layer_1)
    
    fully_connected_with_extra_values = splice(dropout_layer_1,ext_values,axis=0)
    
    fully_connected_layer_2  = Dense(256, init=glorot_uniform())(fully_connected_with_extra_values)
    fully_connected_layer_3  = Dense(128, init=glorot_uniform())(fully_connected_layer_2)
    dropout_layer_2= Dropout(0.5)(fully_connected_layer_3)


    output_layer = Dense(out_dims, init=glorot_uniform(), activation=None)(dropout_layer_2)
    
    return output_layer

def create_model_ext2(input, ext_values, out_dims):
    
    # in VGG style
    #https://www.cs.toronto.edu/~frossard/post/vgg16/
    convolutional_layer_1_1  = Convolution((3,3), 8, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(input)
    convolutional_layer_1_2 = Convolution((5,5), 16, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(convolutional_layer_1_1)
    pooling_layer_1 = MaxPooling((2,2), strides=(2,2))(convolutional_layer_1_2)

    convolutional_layer_2_1 = Convolution((3,3), 16, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_1)
    convolutional_layer_2_2 = Convolution((7,7), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(convolutional_layer_2_1)
    pooling_layer_2 = MaxPooling((2,2), strides=(1,1))(convolutional_layer_2_2)

    convolutional_layer_3_1 = Convolution((3,3), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_2)
    convolutional_layer_3_2 = Convolution((7,7), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(convolutional_layer_3_1)
    pooling_layer_3 = MaxPooling((2,2), strides=(1,1))(convolutional_layer_3_2)

    convolutional_layer_4_1 = Convolution((3,3), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_3)
    pooling_layer_4 = MaxPooling((2,2), strides=(1,1))(convolutional_layer_4_1)

##    
    fully_connected_layer_1  = Dense(512, init=glorot_uniform())(pooling_layer_4)
    dropout_layer_1 = Dropout(0.5)(fully_connected_layer_1)
    
    fully_connected_with_extra_values = splice(dropout_layer_1,ext_values,axis=0)
    
    fully_connected_layer_2  = Dense(256, init=glorot_uniform())(fully_connected_with_extra_values)
    fully_connected_layer_3  = Dense(128, init=glorot_uniform())(fully_connected_layer_2)
    dropout_layer_2= Dropout(0.5)(fully_connected_layer_3)


    output_layer = Dense(out_dims, init=glorot_uniform(), activation=None)(dropout_layer_2)
    
    return output_layer