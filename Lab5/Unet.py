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
from tensorflow.python.keras.layers import Bidirectional, ConvLSTM2D, Reshape
from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
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

def conv_block_3(i, conca_i, Base, acti, bn, drop, img_height, img_width):
    n = Conv2DTranspose(Base, (3, 3), strides=(2, 2), padding='same')(i)
    # reshaping:
    x1 = Reshape(target_shape=(1, np.int32(img_height/16), 
                               np.int32(img_width/16),
                               Base))(conca_i)

    x2 = Reshape(target_shape=(1, np.int32(img_height/16), 
                               np.int32(img_width/16),
                               Base))(n)
    # concatenation:
    n = concatenate([x1, x2], axis=1)
    # Dropout
    n = Dropout(drop)(n) if drop else n
    # LSTM
    n = ConvLSTM2D(int(Base/2), (3, 3), padding='same', return_sequences=False,
                 go_backwards=True)(n)
    o = conv_block_1(n, Base, acti, bn)
    return o

def get_UNet(img_shape, Base, depth, inc_rate, activation, drop, batchnorm, N):
    i = Input(shape=img_shape)
    img_height = img_shape[0]
    img_width = img_shape[1]
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
        n = conv_block_3(n, x_conca[-1-k], Base, activation, batchnorm, drop, 
                         img_height, img_width)
    
    if N>2:
        o = Conv2D(N, (1,1), activation='softmax')(n)
    else:
        o = Conv2D(1, (1,1), activation='sigmoid')(n)
    
    return Model(inputs=i, outputs=o)

