# This file contains different models for different task
import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import Activation, Dropout, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam



# AlexNet Model model for task 1a
def model_task1a(Base, img_ch, img_width, img_height):
    model_task1a = Sequential()

    model_task1a.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1a.add(Activation('relu'))
    model_task1a.add(MaxPooling2D(pool_size=(2, 2)))

    model_task1a.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1a.add(Activation('relu'))
    model_task1a.add(MaxPooling2D(pool_size=(2, 2)))

    model_task1a.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1a.add(Activation('relu'))

    model_task1a.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1a.add(Activation('relu'))

    model_task1a.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1a.add(Activation('relu'))
    model_task1a.add(MaxPooling2D(pool_size=(2, 2)))

    model_task1a.add(Flatten())
    model_task1a.add(Dense(64))
    model_task1a.add(Activation('relu'))

    model_task1a.add(Dense(64))
    model_task1a.add(Activation('relu'))

    model_task1a.add(Dense(1))
    model_task1a.add(Activation('sigmoid'))

    model_task1a.summary()
    return model_task1a


# AlexNet Model model for task 1b with batch normalization
def model_task1b(Base, img_ch, img_width, img_height):
    model_task1b = Sequential()

    model_task1b.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1b.add(BatchNormalization())
    model_task1b.add(Activation('relu'))
    model_task1b.add(MaxPooling2D(pool_size=(2, 2)))

    model_task1b.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1b.add(BatchNormalization())
    model_task1b.add(Activation('relu'))
    model_task1b.add(MaxPooling2D(pool_size=(2, 2)))

    model_task1b.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1b.add(BatchNormalization())
    model_task1b.add(Activation('relu'))

    model_task1b.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1b.add(BatchNormalization())
    model_task1b.add(Activation('relu'))

    model_task1b.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task1b.add(BatchNormalization())
    model_task1b.add(Activation('relu'))
    model_task1b.add(MaxPooling2D(pool_size=(2, 2)))

    model_task1b.add(Flatten())
    model_task1b.add(Dense(64))
    model_task1b.add(Activation('relu'))

    model_task1b.add(Dense(64))
    model_task1b.add(Activation('relu'))

    model_task1b.add(Dense(1))
    model_task1b.add(Activation('sigmoid'))

    model_task1b.summary()
    return model_task1b


# Task 2
# AlexNet Model with dropout and batch normalization
def model_task2(Base, img_ch, img_width, img_height):
    model_task2 = Sequential()

    model_task2.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task2.add(BatchNormalization())
    model_task2.add(Activation('relu'))
    model_task2.add(MaxPooling2D(pool_size=(2, 2)))

    model_task2.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task2.add(BatchNormalization())
    model_task2.add(Activation('relu'))
    model_task2.add(MaxPooling2D(pool_size=(2, 2)))

    model_task2.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task2.add(BatchNormalization())
    model_task2.add(Activation('relu'))
    model_task2.add(Dropout(0.4))

    model_task2.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task2.add(BatchNormalization())
    model_task2.add(Activation('relu'))

    model_task2.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task2.add(BatchNormalization())
    model_task2.add(Activation('relu'))
    model_task2.add(MaxPooling2D(pool_size=(2, 2)))

    model_task2.add(Flatten())
    model_task2.add(Dense(64))
    model_task2.add(Activation('relu'))

    model_task2.add(Dense(64))
    model_task2.add(Activation('relu'))

    model_task2.add(Dense(1))
    model_task2.add(Activation('sigmoid'))

    model_task2.summary()
    return model_task2


# Task 3 (first part)
# AlexNet Model with spatial_dropout and without batch normalization
def model_task3_part1(Base, img_ch, img_width, img_height):
    model_task3_part1 = Sequential()

    model_task3_part1.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part1.add(Activation('relu'))
    model_task3_part1.add(SpatialDropout2D(0.1))
    model_task3_part1.add(MaxPooling2D(pool_size=(2, 2)))

    model_task3_part1.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part1.add(Activation('relu'))
    model_task3_part1.add(SpatialDropout2D(0.4))
    model_task3_part1.add(MaxPooling2D(pool_size=(2, 2)))

    model_task3_part1.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part1.add(Activation('relu'))
    model_task3_part1.add(SpatialDropout2D(0.4))

    model_task3_part1.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part1.add(Activation('relu'))
    model_task3_part1.add(SpatialDropout2D(0.4))

    model_task3_part1.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part1.add(Activation('relu'))
    model_task3_part1.add(SpatialDropout2D(0.4))
    model_task3_part1.add(MaxPooling2D(pool_size=(2, 2)))

    model_task3_part1.add(Flatten())
    model_task3_part1.add(Dense(64))
    model_task3_part1.add(Activation('relu'))

    model_task3_part1.add(Dense(64))
    model_task3_part1.add(Activation('relu'))

    model_task3_part1.add(Dense(1))
    model_task3_part1.add(Activation('sigmoid'))

    model_task3_part1.summary()
    return model_task3_part1


# Task 3 (second part)
# AlexNet Model with spatial_dropout and without batch normalization
def model_task3_part2(Base, img_ch, img_width, img_height):
    model_task3_part2 = Sequential()

    model_task3_part2.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part2.add(Activation('relu'))
    model_task3_part2.add(MaxPooling2D(pool_size=(2, 2)))

    model_task3_part2.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part2.add(Activation('relu'))
    model_task3_part2.add(MaxPooling2D(pool_size=(2, 2)))

    model_task3_part2.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part2.add(Activation('relu'))

    model_task3_part2.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part2.add(Activation('relu'))

    model_task3_part2.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model_task3_part2.add(Activation('relu'))
    model_task3_part2.add(MaxPooling2D(pool_size=(2, 2)))

    model_task3_part2.add(Flatten())
    model_task3_part2.add(Dense(64))
    model_task3_part2.add(Activation('relu'))

    model_task3_part2.add(Dense(64))
    model_task3_part2.add(Activation('relu'))

    model_task3_part2.add(Dense(1))
    model_task3_part2.add(Activation('sigmoid'))

    model_task3_part2.summary()
    return model_task3_part2


# Task4 improved VGG16 model
def model_VGG16(Base, img_ch, img_width, img_height):
    model_VGG16 = Sequential()
    model_VGG16.add(Conv2D(Base, input_shape=(img_width, img_height, img_ch), kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model_VGG16.add(Conv2D(Base, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(BatchNormalization())
    model_VGG16.add(SpatialDropout2D(0.1))
    model_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16.add(Conv2D(Base * 2, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(Conv2D(Base * 2, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(BatchNormalization())
    model_VGG16.add(SpatialDropout2D(0.4))
    model_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16.add(Conv2D(Base * 4, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(Conv2D(Base * 4, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(Conv2D(Base * 4, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(BatchNormalization())
    model_VGG16.add(SpatialDropout2D(0.4))
    model_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(BatchNormalization())
    model_VGG16.add(SpatialDropout2D(0.4))
    model_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16.add(BatchNormalization())
    model_VGG16.add(SpatialDropout2D(0.4))
    model_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16.add(Flatten())
    model_VGG16.add(Dense(Base * 64, activation='relu'))
    model_VGG16.add(Dense(Base * 64, activation='relu'))
    model_VGG16.add(Dense(10, activation='softmax'))

    model_VGG16.summary()
    return model_VGG16
