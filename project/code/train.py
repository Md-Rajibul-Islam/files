#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
from tensorflow.keras import utils
from ResNet18 import resnet_18
from plot import plot_learning_curve, plot_validation_metric


# Data augmentation
train_datagen = ImageDataGenerator(rescale=1 / 255.0,
                              rotation_range=20,
                              zoom_range=0.05,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.05,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode="nearest")

val_datagen = ImageDataGenerator(rescale=1 / 255.0)

# Initialize the training and validation generator
Batch_size = 16
train_gen = train_datagen.flow_from_directory(
    '/dl_data/tra/',
    class_mode='categorical',
    target_size=(224, 224),
    color_mode='rgb',
    shuffle=True,
    batch_size=Batch_size,
    seed=42)


val_gen = val_datagen.flow_from_directory(
    '/dl_data/val/',
    class_mode='categorical',
    target_size=(224, 224),
    color_mode='rgb',
    shuffle=True,
    batch_size=Batch_size,
    seed=42)


# Calculate class weight 
train = list(paths.list_images('/dl_data/tra'))
y_train = [int(name.split(os.path.sep)[-2]) for name in train]
y_train = utils.to_categorical(y_train)
class_weight = np.max(y_train.sum(axis=0)) / y_train.sum(axis=0)

# Create and train the model
model = resnet_18(n_class=2)
epochs = 150
lr = 1e-5

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr), 
              metrics=['accuracy'])

History = model.fit_generator(
    train_gen,
    steps_per_epoch=train_gen.n//Batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=val_gen,
    validation_steps=val_gen.n//Batch_size,
    class_weight=class_weight)

# Plot learning curves
plot_learning_curve(History, 'loss')
plot_validation_metric(History, 'metrics')

# Save the trained model
model.save('model.h5')
print("Model saved")

