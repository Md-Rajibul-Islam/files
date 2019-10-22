#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


def conv_bn_relu(i, base, kernel_size, strides=(1, 1), padding='same'):
    
    n = Conv2D(base, kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_regularizer=regularizers.l2(0.0001))(i)
    n = BatchNormalization()(n)
    n = Activation('relu')(n)
    
    return n

def _bn_relu_conv(i, base, kernel_size, strides=(1, 1), padding='same'):
    
    n = BatchNormalization()(i)
    n = Activation('relu')(n)
    n = Conv2D(base, kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_regularizer=regularizers.l2(1e-4))(n)
    return n
    
    
def shortcut(i, residual):
 
    input_shape = K.int_shape(i)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
 
    identity = i
    
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3],
                           kernel_size=(1, 1),
                           strides=(stride_width, stride_height),
                           padding='valid',
                           kernel_regularizer=regularizers.l2(1e-4))(i)
 
    return add([identity, residual])

def basic_block(i, base, strides, is_first_block_of_first_layer=False):
    
    if is_first_block_of_first_layer:
        conv1 = Conv2D(base, kernel_size=(3, 3),strides=(1, 1), padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(1e-4))(i)
    else:
        conv1 = _bn_relu_conv(i, base, kernel_size=(3, 3), strides=strides)

    residual = _bn_relu_conv(conv1, base, kernel_size=(3, 3))

    return shortcut(i, residual)

    

def residual_block(i, base, repetitions, is_first_layer=False):
   
    n=i
    for j in range(repetitions):
        if j == 0 and not is_first_layer:
            strides = (2, 2)
        else:
            strides = (1, 1)
        
        n = basic_block(n, base, strides, 
                        is_first_block_of_first_layer=(is_first_layer and j == 0))
        
    return n



def resnet_18(n_class, input_shape=(224,224,3)):
    '''ResNet has 5 stacks '''
    i = Input(shape=input_shape)
    base = 64
    '''Stack 1'''
    conv1 = conv_bn_relu(i, base=base, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
    
    '''Stack 2-5'''
    stack = pool1
    for j in range(4):
        stack = residual_block(stack, base, repetitions=2, is_first_layer=(j==0))
        base *=2
     
    n = BatchNormalization()(stack)
    n = Activation('relu')(n)
    
    n_shape = K.int_shape(n)
    pool2 = AveragePooling2D(pool_size=(n_shape[1], n_shape[2]),strides=(1, 1))(n)
    f = Flatten()(pool2)
    o = Dense(n_class, activation='softmax')(f)
 
    model = Model(inputs=i, outputs=o)
    #model.summary()
 
    return model
