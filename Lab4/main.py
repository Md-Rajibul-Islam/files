#!/usr/bin/env python
# coding: utf-8

from dataloader_kfold import gen_list, shuffle, split_list, get_len, read_data, binary_mask
from Unet import get_UNet
from plot import plot_learning_curve, plot_validation_metric
from metrics import dice_coef_loss, dice_coef, precision, recall, weighted_loss
from augmentation import combine_generator, generator_with_weights
import SimpleITK as sitk
from itertools import chain

path = '/Lab1/Lab3/MRI/'
img_h, img_w = 240, 240
Mask = sorted(gen_list(path, 'Mask'))
Img = sorted(gen_list(path,'Image'))
Mask, Img = shuffle(Mask, Img)



''' Task 1 ''' 

k=3 # k=5
Mask = split_list(Mask,k)
Img = split_list(Img,k)

for i in range(k):
    Mask_val = list(Mask[i])
    Mask_train =list(chain.from_iterable(Mask[:i] + Mask[i+1:])) 
    Img_val = list(Img[i])
    Img_train =list(chain.from_iterable(Img[:i] + Img[i+1:]))
    
    Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w)
    Mask_val =read_data(path+'Mask/', Mask_val, img_h, img_w)
    Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
    Img_val = read_data(path+'Image/', Img_val, img_h, img_w)
    
    model = get_UNet(img_shape=(img_h, img_w, 1), Base=16, depth=4, inc_rate=2, 
                 activation='relu', drop=0, batchnorm=True, N=2, weight_use=False)

    model.compile(optimizer=Adam(lr=1e-5), loss=[dice_coef_loss], 
                  metrics=[dice_coef,precision,recall])

    History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=1, 
                        validation_data=(Img_val, Mask_val))

    plot_learning_curve(History, 'Task1_k={0}_loss_{1}_'.format(k,i+1))
    plot_validation_metric(History, 'Task1_k={0}_metrics_{1}_'.format(k,i+1))
    

''' Task 2 ''' 

k=3 # k=5
Mask = split_list(Mask,k)
Img = split_list(Img,k)

radius = 2
weight_strength=1
batch_size =8

for i in range(k):
    Mask_val = list(Mask[i])
    Mask_train =list(chain.from_iterable(Mask[:i] + Mask[i+1:])) 
    Img_val = list(Img[i])
    Img_train =list(chain.from_iterable(Img[:i] + Img[i+1:]))
    
    Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w, binary=True)
    Mask_val =read_data(path+'Mask/', Mask_val, img_h, img_w, binary=True)
    Img_train = read_data(path+'Image/', Img_train, img_h, img_w, binary=True)
    Img_val = read_data(path+'Image/', Img_val, img_h, img_w, binary=True)
    
    weight_train = binary_mask(Mask_train, radius)
    weight_val = binary_mask(Mask_val, radius) 
    
    model, loss_weights = get_UNet(img_shape=(img_h, img_w ,1), Base=16, depth=4, 
                                   inc_rate=2, activation='relu', drop=0,
                                   batchnorm=True, N=2, weight_use=True)
    
    # Compile model with appropriate loss function
    model.compile(optimizer=Adam(lr=1e-5),
                  loss=weighted_loss(loss_weights, weight_strength),
                  metrics=[dice_coef,precision,recall])
    
    train_generator = generator_with_weights(Img_train, Mask_train, 
                                             weight_train, batch_size)
    
    # Training
    History = model.fit_generator(train_generator, steps_per_epoch =
                                  np.ceil(float(len(Img_train))/float(batch_size)),
                                  epochs = 150,
                                  verbose=1, max_queue_size=1, 
                                  validation_steps=len(Img_val),
                                  validation_data=([Img_val, weight_val],Mask_val),
                                  shuffle=True, class_weight='auto')

    
    plot_learning_curve(History, 'Task2_k={0}_loss_{1}'.format(k,i+1))
    plot_validation_metric(History, 'Task2_k={0}_metrics_{1}'.format(k,i+1))
    

''' Task 3 '''

k=3 # k=5
Mask = split_list(Mask,k)
Img = split_list(Img,k)

Batch_size=8
model_predictions = np.zeros((len(Mask),img_h, img_w, 1))
dataPath = 'task3/'

for i in range(k):
    Mask_val = list(Mask[i])
    Mask_train =list(chain.from_iterable(Mask[:i] + Mask[i+1:])) 
    Img_val = list(Img[i])
    Img_train =list(chain.from_iterable(Img[:i] + Img[i+1:]))
    
    Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w, binary=True)
    Mask_val =read_data(path+'Mask/', Mask_val, img_h, img_w, binary=True)
    Img_train = read_data(path+'Image/', Img_train, img_h, img_w, binary=True)
    Img_val = read_data(path+'Image/', Img_val, img_h, img_w, binary=True)
    
    if i == 0:
        y_pred_train = np.ones((len(Img_train), img_h, img_w, 1),dtype=np.float32)/2
        y_pred_val = np.ones((len(Img_val), img_h, img_w, 1),dtype=np.float32)/2
        
    else:
        y_pred = np.load(dataPath + str(i - 1) + '.npy')
        y_pred_train = np.vstack((y_pred[:get_len(Mask[:i])], 
                                  y_pred[-get_len(Mask[i+1:]):]))
        y_pred_val = y_pred[get_len(Mask[:i]):(get_len(Mask[:i])+len(Mask[i]))]

    # Concatenate image data and posterior probabilities:
    Img_train = np.concatenate((Img_train, y_pred_train), axis=-1)
    Img_val = np.concatenate((Img_val, y_pred_val), axis=-1)
    
    model = get_UNet(img_shape=(img_h, img_w, 2), Base=16, depth=4, inc_rate=2, 
                 activation='relu', drop=0, batchnorm=True, N=2, weight_use=False)

    model.compile(optimizer=Adam(lr=1e-5), loss=[dice_coef_loss], 
                  metrics=[dice_coef,precision,recall])

    History = model.fit(Img_train, Mask_train, batch_size=Batch_size, epochs=150, 
                        verbose=1, validation_data=(Img_val, Mask_val))
    
    val_predictions = model.predict(Img_val, batch_size=int(Batch_size / 2))
    model_predictions[get_len(Mask[:i]):
                       (get_len(Mask[:i])+len(Mask[i]))] = val_predictions
    np.save(dataPath + str(i) + '.npy', model_predictions)
    
    plot_learning_curve(History, 'Task3_k={0}_loss_{1}_'.format(k,i+1))
    plot_validation_metric(History, 'Task3_k={0}_metrics_{1}_'.format(k,i+1))