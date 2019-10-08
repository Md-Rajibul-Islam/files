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

def get_LSTM(num_layers, units, batch_size, input_size, 
             input_dimension, Bi,drop):
    
    i = Input(batch_shape=(batch_size, input_size, input_dimension))
    n = i
    for j in range(num_layers):
        if j == 0 and Bi == True:
            n = Bidirectional(LSTM(units, return_sequences=True, 
                                   stateful=True))(n)
        if j == num_layers-1:
            n = LSTM(units, return_sequences=False, stateful=True)(n)
        else:
            n = LSTM(units, return_sequences=True, stateful=True)(n)    
        n = Dropout(drop)(n) if drop else n
    if input_size != None:
        n = Flatten()(n)
    o = Dense(1)(n)
    return Model(inputs=i, outputs=o)

