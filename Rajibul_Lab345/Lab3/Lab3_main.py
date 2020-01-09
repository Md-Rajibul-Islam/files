from model_UNet import get_unet, get_unet_multi_organs
from similarity_metrices import dice_coef_loss, dice_coef, precision, recall
from data_loader import shuffle_split, gen_list, read_data, read_mask_onehot_encoding
from plot import plot_learning_curve, plot_validation_metric

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator

#from augmentation_generator import image_train_datagen, mask_train_datagen
#from augmentation_generator import image_validation_datagen, mask_validation_datagen

from augmentation_generator import aug
import tensorflow as tf


path = '/Lab1/Lab3/X_ray/'
Img = gen_list(path, 'Image')
Mask = gen_list(path, 'Mask')
img_h, img_w = 256, 256


Img_train, Img_validation, Mask_train, Mask_validation = shuffle_split(Img, Mask, 80)  # Image and mask distribution

Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w)
Mask_validation = read_data(path+'Mask/', Mask_validation, img_h, img_w)
Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
Img_validation = read_data(path+'Image/', Img_validation, img_h, img_w)


# calling the model
model = get_unet(input_img=(256, 256, 1), n_filters=16, kernel_size=3, dropout=0.5, batchnorm=True)

# Task 1a
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=[dice_coef])

History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=2,
                    validation_data=(Img_validation, Mask_validation))
plot_learning_curve(History, 'Lab3_task1a')
print('Task1a done !')


# Task 1b
model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])

History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=2,
                    validation_data=(Img_validation, Mask_validation))

plot_learning_curve(History, 'Lab3_task1b')
print('Task 1b done !')


# Task 2a
model = get_unet(input_img=(256, 256, 1), n_filters=16, kernel_size=3, dropout=0.5, batchnorm=False)

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=[dice_coef])
History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=2,
                    validation_data=(Img_validation, Mask_validation))

plot_learning_curve(History, 'Lab3_task2a')
print('Task2a done !')


# Task 2b
model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=2,
                    validation_data=(Img_validation, Mask_validation))

plot_learning_curve(History, 'Lab3_task2b')
print('Task 2b done !')



# Task 3
model = get_unet(input_img=(256, 256, 1), n_filters=32, kernel_size=3, dropout=0.5, batchnorm=True)

model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])

History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=2,
                    validation_data=(Img_validation, Mask_validation))


plot_learning_curve(History, 'Lab3_task3_dice_coef_loss')
print('Task 3 done !')



# Task 4
model = get_unet(input_img=(256, 256, 1), n_filters=32, kernel_size=3, dropout=0.5, batchnorm=True)
model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])

BS = 8
EPOCHS = 150
History = model.fit_generator(aug.flow(Img_train, Mask_train, batch_size=BS),
                              validation_data=(Img_validation, Mask_validation), steps_per_epoch=len(Img_train) // BS,
                              epochs=EPOCHS)

plot_learning_curve(History, 'Lab3_task4_dice_coef_loss')
print('Task 4 done !')


# Task 5
path = '/Lab1/Lab3/CT/'
img_h, img_w = 256, 256
Img = gen_list(path, 'Image')
Mask = gen_list(path, 'Mask')


Mask_train, Mask_validation, Img_train, Img_validation = shuffle_split(Mask, Img, 80)  # Image and mask distribution

Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w)
Mask_validation = read_data(path+'Mask/', Mask_validation, img_h, img_w)

Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
Img_validation = read_data(path+'Image/', Img_validation, img_h, img_w)

model = get_unet(input_img=(256, 256, 1), n_filters=16, kernel_size=3, dropout=0.5, batchnorm=True)


# Task 5a1
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=[dice_coef])
History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=2,
                    validation_data=(Img_validation, Mask_validation))

plot_learning_curve(History, 'Lab3_task5a1')
print('Task5a1 done !')


# Task 5a2
model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])

History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=2,
                    validation_data=(Img_validation, Mask_validation))

plot_learning_curve(History, 'Lab3_task5a2')
print('Task 5a2 done !')


# Task 5b, loss = 'binary_crossentropy'
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=[dice_coef, precision, recall])
BS = 8
EPOCHS = 150
History = model.fit_generator(aug.flow(Img_train, Mask_train, batch_size=BS),
                              validation_data=(Img_validation, Mask_validation), steps_per_epoch=len(Img_train) // BS,
                              epochs=EPOCHS)

plot_learning_curve(History, 'Lab3_task5b1_learning_curve')
plot_validation_metric(History, 'Lab3_task5b1_metrics')
print('Task 5b1 done !')


# Task 5b2, loss=[dice_coef_loss]
model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, precision, recall])

History = model.fit_generator(aug.flow(Img_train, Mask_train, batch_size=BS),
                              validation_data=(Img_validation, Mask_validation), steps_per_epoch=len(Img_train) // BS,
                              epochs=EPOCHS)

plot_learning_curve(History, 'Lab3_task5b2_learning_curve')
plot_validation_metric(History, 'Lab3_task5b2_metrics')
print('Task 5b2 done !')


# Task 6
path = '/Lab1/Lab3/CT/'
img_h, img_w = 256, 256
Img = gen_list(path, 'Image')
Mask = gen_list(path, 'Mask')


Mask_train, Mask_validation, Img_train, Img_validation = shuffle_split(Mask, Img, 80)  # Image and mask distribution


Mask_train = read_mask_onehot_encoding(path+'Mask/', Mask_train, img_h, img_w)
Mask_validation = read_mask_onehot_encoding(path+'Mask/', Mask_validation, img_h, img_w)

Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
Img_validation = read_data(path+'Image/', Img_validation, img_h, img_w)


model = get_unet_multi_organs(input_img=(256, 256, 1), n_filters=16, kernel_size=3, N=3, dropout=0.5, batchnorm=True)
BS = 8
EPOCHS = 100
model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, precision, recall])
History = model.fit_generator(aug.flow(Img_train, Mask_train, batch_size=BS),
                              validation_data=(Img_validation, Mask_validation), steps_per_epoch=len(Img_train) // BS,
                              epochs=EPOCHS)

plot_learning_curve(History, 'Lab3_task6_learning_curve')
plot_validation_metric(History, 'Lab3_task6_metrics')
print('Task 6 done !')


# Task 7
path = '/Lab1/Lab3/MRI/'
img_h, img_w = 240, 240
Img = gen_list(path, 'Image')
Mask = gen_list(path, 'Mask')


Mask_train, Mask_validation, Img_train, Img_validation = shuffle_split(Mask, Img, 80)  # Image and mask distribution


Mask_train = read_mask_onehot_encoding(path+'Mask/', Mask_train, img_h, img_w)
Mask_validation = read_mask_onehot_encoding(path+'Mask/', Mask_validation, img_h, img_w)

Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
Img_validation = read_data(path+'Image/', Img_validation, img_h, img_w)

model = get_unet(input_img=(240, 240, 1), n_filters=16, kernel_size=3, dropout=0.5, batchnorm=True)

model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, precision, recall])
History = model.fit(Img_train, Mask_train, batch_size=4, epochs=100, verbose=1,
                    validation_data=(Img_validation, Mask_validation))


plot_learning_curve(History, 'Lab3_task7_learning_curve')
plot_validation_metric(History, 'Lab3_task7_metrics')
print('Task 7 done !')

