#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Input, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import SimpleITK as sitk
import math


def conv_block_1(i, Base, acti, bn):
    n = Conv2D(Base, (3,3), padding='same')(i)
    n = BatchNormalization()(n) if bn else n
    n = Activation(acti)(n)
    n = Conv2D(Base, (3,3), padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    o = Activation(acti)(n)
    return o

def conv_block_2(i, Base, acti, bn, drop):
    n = MaxPooling2D(pool_size=(2, 2))(i)
    n = Dropout(drop)(n) if drop else n
    o = conv_block_1(n, Base, acti, bn)
    return o

def conv_block_3(i, conca_i, Base, acti, bn, drop):
    n = Conv2DTranspose(Base, (2, 2), strides=(2, 2), padding='same')(i)
    n = concatenate([n, conca_i], axis=3)
    n = Dropout(drop)(n) if drop else n
    o = conv_block_1(n, Base, acti, bn)
    return o

def get_UNet(img_shape, Base, depth, inc_rate, activation, 
             drop, batchnorm, N, weight_use):
    i = Input(shape=img_shape)
    img_height = img_shape[0]
    img_width = img_shape[1]
    loss_weights = Input((img_height, img_width, 1))
    x_conca = []
    n = conv_block_1(i, Base, activation, batchnorm)
    x_conca.append(n)
    for k in range(depth):
        Base = Base*inc_rate
        n = conv_block_2(n, Base, activation, batchnorm, drop)
        if k < (depth-1):
            x_conca.append(n)
    for k in range(depth):
        Base = Base//inc_rate
        img_height = img_height*inc_rate
        img_width = img_width*inc_rate
        n = conv_block_3(n, x_conca[-1-k], Base, activation, batchnorm, drop)
    
    if N>2:
        o = Conv2D(N, (1,1), activation='softmax')(n)
    else:
        o = Conv2D(1, (1,1), activation='sigmoid')(n)
    if weight_use:
        return Model(inputs=[i, loss_weights], outputs=o), loss_weights
    else:
        return Model(inputs=i, outputs=o)
