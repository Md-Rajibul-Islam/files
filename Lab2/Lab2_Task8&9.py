#!/usr/bin/env python
# coding: utf-8

# In[1]:
from plot_lab2 import plot_learning_curve

# In[16]:


# Task 8,9
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import applications
import numpy as np
import matplotlib.pyplot as plt


def get_length(Path, Pattern):
    Length =  len(os.listdir(os.path.join(Path, Pattern)))
    return Length

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.summary()
    

# parameters (TODO)
#train_data_dir = '/Lab1/Lab2/Bone/train/'
#validation_data_dir = '/Lab1/Lab2/Bone/validation/'
train_data_dir = '/Lab1/Lab2/Skin/train/'
validation_data_dir = '/Lab1/Lab2/Skin/validation/'
img_width, img_height, img_ch = 224, 224, 3
epochs = 150
batch_size = 8
LR = 0.00001
# number of data for each class
Len_C1_Train = get_length(train_data_dir,'Mel')
Len_C2_Train = get_length(train_data_dir,'Nevi')
Len_C1_Val = get_length(validation_data_dir,'Mel')
Len_C2_Val = get_length(validation_data_dir,'Nevi')
'''Len_C1_Train = get_length(train_data_dir,'AFF')
Len_C2_Train = get_length(train_data_dir,'NFF')
Len_C1_Val = get_length(validation_data_dir,'AFF')
Len_C2_Val = get_length(validation_data_dir,'NFF')'''

# loading the pre-trained model
model = applications.VGG16(include_top=False, weights='imagenet')
#model.summary()

# Feature extraction from pretrained VGG (training data)
datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

features_train = model.predict_generator(
        train_generator,
        (Len_C1_Train+Len_C2_Train) // batch_size, max_queue_size=1)


# To DO: Feature extraction from pretrained VGG (validation data)
datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

features_validation = model.predict_generator(
        validation_generator,
        (Len_C1_Val+Len_C2_Val) // batch_size, max_queue_size=1)


# training a small MLP with extracted features from the pre-trained model
train_data = features_train
train_labels = np.array([0] * int(Len_C1_Train) + [1] * int(Len_C2_Train))

validation_data = features_validation
validation_labels = np.array([0] * int(Len_C1_Val) + [1] * int(Len_C2_Val))


# In[17]:


# TODO: Building the MLP model
def model(img_width, img_height, img_ch):
    model = Sequential()
    model.add(Flatten(input_shape=(img_width, img_height, img_ch)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    return model

model_MLP=model(7, 7, 512)


# In[18]:


# TODO: Compile and train the model, plot learning curves
model_MLP.compile(loss='binary_crossentropy',
              optimizer = Adam(lr=LR),
              metrics=['accuracy'])

History = model_MLP.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=2, 
                    validation_data=(validation_data, validation_labels))

#plot_learning_curve(History, 'Task8_Bone')
plot_learning_curve(History, 'Task9_Skin')

# In[3]:





# In[ ]:




