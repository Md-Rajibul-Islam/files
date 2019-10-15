# Lab2: Task6 and Task7
import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Activation, Dropout, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

from plot_lab2 import plot_learning_curve_subplot


TRAIN_DIR = '/Lab1/Lab2/Skin/train/'
VAL_DIR = '/Lab1/Lab2/Skin/validation/'


from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_h, img_w = 128, 128

Base = 32  # we are recommended 64
img_ch = 3
lose = 'mean_squared_error'
optimizer = Adam(lr=0.00001)
metrics = ['binary_accuracy']

batch_size = 8
epochs = 80  # 80


# AlexNet Model with drop out and batch normalization
def model(Base, img_ch, img_width, img_height):
    model = Sequential()

    model.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=Base * 4, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=Base * 2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model

model_AlexNet_task_6 = model


Mymodel=model_AlexNet_task_6

img_height, img_width = 128, 128


train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, rescale=1. / 255,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(128, 128), batch_size=batch_size,
                                                    class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(128, 128), batch_size=batch_size,
                                                class_mode='binary')


model = Mymodel(Base, img_ch, img_width= 128, img_height=128)
model.compile(loss=lose, optimizer=optimizer, metrics=['binary_accuracy'])
History = model.fit_generator(train_generator, steps_per_epoch=len(train_generator), verbose=2,
                              validation_data=val_generator, validation_steps=len(val_generator), epochs=epochs)

plot_learning_curve_subplot(History, 'Lab2_Task6_new')




# Task 7 Apply data augmentation on VGG Model For Skin Dataset
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

from plot_lab2 import subplot_value_accuracy
from model_lab2 import model_VGG16

model_VGG_augmentation = model_VGG16(64, 3, 128, 128) # This model from task 4
Epochs = 80  # 80 epochs
LR = 0.00001
batch_size = 8


TRAIN_DIR = '/Lab1/Lab2/Skin/train/'
VAL_DIR = '/Lab1/Lab2/Skin/validation/'

train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, rescale=1. / 255,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(128, 128), batch_size=batch_size,
                                                    class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(128, 128), batch_size=batch_size,
                                                class_mode='binary')

model_VGG_augmentation.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
History = model_VGG_augmentation.fit_generator(train_generator,
                                               steps_per_epoch=(len(train_generator) / batch_size),
                                               validation_data=val_generator,
                                               validation_steps=(len(val_generator) / batch_size),
                                               epochs=Epochs)
subplot_value_accuracy(History, 'Lab2_Task7_Skin_data_new')

# Task 7 VGG Model For Bone Dataset
TRAIN_DIR = '/Lab1/Lab2/Bone/train/'
VAL_DIR = '/Lab1/Lab2/Bone/validation/'

train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, rescale=1. / 255,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(128, 128), batch_size=batch_size,
                                                    class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(128, 128), batch_size=batch_size,
                                                class_mode='binary')

model_VGG_augmentation.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
History = model_VGG_augmentation.fit_generator(train_generator,
                                               steps_per_epoch=(len(train_generator) / batch_size),
                                               validation_data=val_generator,
                                               validation_steps=(len(val_generator) / batch_size),
                                               epochs=Epochs)
subplot_value_accuracy(History, 'Lab2_Task7_Bone_data_new')
