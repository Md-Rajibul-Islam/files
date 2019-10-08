import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)

import os
from tensorflow.keras.optimizers import Adam

from data_lab2 import get_train_test_data
from plot_lab2 import plot_learning_curve
from plot_lab2 import plot_learning_curve_subplot

from model_lab2 import model_task1a
from model_lab2 import model_task1b
from model_lab2 import model_task2
from model_lab2 import model_task3_part1
from model_lab2 import model_task3_part2
from model_lab2 import model_VGG16


if __name__ == "__main__":

    img_w = 128    # Setting the width and heights of the images
    img_h = 128
    data_path = '/Lab1/Skin/'           # Path to data root. Inside this path,
                                                        #two subfolder are placed one for train data and one for test data.
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')

    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)

    x_train, x_test, y_train, y_test = get_train_test_data(train_data_path, test_data_path, train_list, test_list)

# --------------------Task 1a --------------------

# Compiling parameters
lose = 'mean_squared_error'
optimizer = Adam(lr=0.0001)

# History parameters
batch_size = 8
epochs = 50   # change to 50

model = model_task1a(32, 1, 128, 128)
# Base = 8, img_ch = 1, img_width = 128, img_height = 128

model.compile(loss=lose, optimizer=optimizer,
                      metrics=['binary_accuracy'])
History = model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs,
                            verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task1a_new')

print('--------------   End Task1a--------------')


# Compiling parameters
lose = 'mean_squared_error'
optimizer = Adam(lr=0.0001)

# History parameters
batch_size = 8
epochs = 50  # change to 50

model = model_task1b(32, 1, 128, 128)
model.compile(loss=lose, optimizer=optimizer,
                      metrics=['binary_accuracy'])
History = model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs,
                            verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task1b_new')

print('--------------   End Task1b--------------')


# Task1c, We use the model from task1a and task1b with the following parameters
# Compiling parameters
lose = 'mean_squared_error'
optimizer = Adam(lr=0.00001)

# History parameters
batch_size = 8
epochs = 80   # change to 80

model = model_task1a(32, 1, 128, 128)
model.compile(loss=lose, optimizer=optimizer,
                      metrics=['binary_accuracy'])
History = model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs,
                            verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task1c_without_batch_normalization_new')


model = model_task1b(32, 1, 128, 128)
model.compile(loss=lose, optimizer=optimizer,
                      metrics=['binary_accuracy'])
History = model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs,
                            verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task1c_with_batch_normalization_new')

print('--------------   End Task1c  --------------')

# Task 2
# parameters setting
lose = 'mean_squared_error'
optimizer = Adam(lr=0.00001)
batch_size = 8
epochs = 80   # change to 80

model = model_task2(32, 1, 128, 128)
model.compile(loss=lose, optimizer=optimizer,
                      metrics=['binary_accuracy'])
History = model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs,
                            verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task2_with_batch_normalization_new')

print('--------------   End Task2  --------------')

# Setting parameters for Task 3
# Parameters setting
lose = 'mean_squared_error'
optimizer = Adam(lr=0.00001)

batch_size = 8
epochs = 150

model = model_task3_part1(64, 1, 128, 128)
model.compile(loss=lose, optimizer=optimizer,
                      metrics=['binary_accuracy'])
History = model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs,
                            verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task3_part1_new')


model = model_task3_part2(64, 1, 128, 128)
model.compile(loss=lose, optimizer=optimizer,
                      metrics=['binary_accuracy'])
History = model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs,
                            verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task3_part2_new')

print('--------------   End Task3 --------------')


# Task 4, improved VGG16 model
# Parameters
batch_size = 32
epochs = 250   # make 250 epochs

model_VGG = model_VGG16(32, 1, 128, 128)
model_VGG.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['binary_accuracy'])
History = model_VGG.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task4_Skin_data_new')


# Reading Bone data
from data_lab2_bone import get_train_test_data

img_w, img_h = 128, 128  # Setting the width and heights of the images
data_path = '/Lab1/Bone/'  # Path to data root. Inside this path,
# two subfolder are placed one for train data and one for test data.


train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

train_list = os.listdir(train_data_path)
test_list = os.listdir(test_data_path)

x_train, x_test, y_train, y_test = get_train_test_data(
    train_data_path, test_data_path,
    train_list, test_list)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')


model = model_VGG16(32, 1, 128, 128)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])

# set epochs = 300
History = model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=2, validation_data=(x_test, y_test))
plot_learning_curve_subplot(History, 'Lab2_Task4_Bone_data_new')

print('---------    End Task 4 -----------')


