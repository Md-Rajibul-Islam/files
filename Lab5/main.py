#!/usr/bin/env python
# coding: utf-8

from dataloader import gen_list, shuffle_split, read_data
from Unet import get_UNet
from LSTMnetwork import get_LSTM
from plot import plot_learning_curve, plot_validation_metric_1, plot_validation_metric_2, plot_validation_metric_3
from metrics import dice_coef_loss, dice_coef, precision, recall
from augmentation import XYaugmentGenerator
import SimpleITK as sitk
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib
from tensorflow.keras.utils import Sequence

''' Task 1 ''' 

# load csv files: 
dataset_train = pd.read_csv('/Lab1/Lab5/train_data_stock.csv') 
dataset_val = pd.read_csv('/Lab1/Lab5/val_data_stock.csv')

# reverse data so that they go from oldest to newest: 
dataset_train = dataset_train.iloc[::-1] 
dataset_val = dataset_val.iloc[::-1] 

# concatenate training and test datasets: 
dataset_total = pd.concat((dataset_train['Open'], dataset_val['Open']), axis=0) 

# select the values from the “Open” column as the variables to be predicted: 
training_set = dataset_train.iloc[:, 1:2].values 
val_set = dataset_val.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range=(0, 1)) 
training_set_scaled = sc.fit_transform(training_set)

# split training data into T time steps: 
X_train = [] 
y_train = [] 
T=60

for i in range(T, len(training_set)): 
    X_train.append(training_set_scaled[i-T:i, 0]) 
    y_train.append(training_set_scaled[i, 0]) 
    
X_train, y_train = np.array(X_train), np.array(y_train)

# normalize the validation set according to the normalization applied to the training set: 
inputs = dataset_total[len(dataset_total) - len(dataset_val) - 60:].values 
inputs = inputs.reshape(-1, 1) 
inputs = sc.transform(inputs)

# split validation data into T time steps: 
X_val = []
for i in range(T, T + len(val_set)): 
    X_val.append(inputs[i-T:i, 0]) 
X_val = np.array(X_val) 
y_val = sc.transform(val_set) 
# reshape to 3D array (format needed by LSTMs -> number of samples, timesteps, input dimension) 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

units = 20 
# units = 40
# units = 60
model = get_LSTM(num_layers=4, units=units, batch_size=16, input_size=X_train.shape[1], 
             input_dimension=1, Bi=False, drop=0.2)
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
History = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, 
                    validation_data=(X_val, y_val))

plot_learning_curve(History, 'Task1_units={0}_loss'.format(units))
plot_validation_metric_1(History, 'Task1_units={0}_metrics'.format(units))

''' Task 2 ''' 

def load_streamlines(dataPath, subject_ids, bundles, n_tracts_per_bundle):
    X = []
    y = []
    for i in range(len(subject_ids)):
        for c in range((len(bundles))):
            filename = dataPath + subject_ids[i] + '/' + bundles[c] + '.trk'
            tfile = nib.streamlines.load(filename)
            streamlines = tfile.streamlines
            
            n_tracts_total = len(streamlines)
            ix_tracts = np.random.choice(range(n_tracts_total),
                                         n_tracts_per_bundle, replace=False)
            streamlines_data = streamlines.data
            streamlines_offsets = streamlines._offsets
            
            for j in range(n_tracts_per_bundle):
                ix_j = ix_tracts[j]
                offset_start = streamlines_offsets[ix_j]
                if ix_j < (n_tracts_total - 1):
                    offset_end = streamlines_offsets[ix_j + 1]
                    streamline_j = streamlines_data[offset_start:offset_end]
                else:
                    streamline_j = streamlines_data[offset_start:]
                X.append(np.asarray(streamline_j))
                y.append(c)
    return X, y

dataPath = '/Lab1/Lab5/HCP_lab/' 
train_subjects_list = ['613538', '599671', '599469']
val_subjects_list = ['601127']
bundles_list = ['CST_left', 'CST_right']
n_tracts_per_bundle = 20
X_train, y_train = load_streamlines(dataPath, train_subjects_list,
                                    bundles_list, n_tracts_per_bundle)
X_val, y_val = load_streamlines(dataPath, val_subjects_list, bundles_list,
                                n_tracts_per_bundle)


class MyBatchGenerator(Sequence):
    
    def __init__(self, X, y, batch_size=1, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))
    
    def __getitem__(self, index):
        return self.__data_generation(index)
    
    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, 1))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb
    
units = 10 #units = 8,5,2
batch_size = 1
model = get_LSTM(num_layers=4, units=units, batch_size=batch_size, input_size=None, 
             input_dimension=3, Bi=True, drop=0.2)

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', 
                  metrics=['accuracy'])

History = model.fit_generator(MyBatchGenerator(X_train, y_train, batch_size=1), 
                    epochs=50, 
                    validation_data=MyBatchGenerator(X_val, y_val,batch_size=1), 
                    validation_steps=len(X_val))

plot_learning_curve(History, 'Task2_units={0}_loss'.format(units))
plot_validation_metric_2(History, 'Task2_units={0}_metrics'.format(units))

''' Task 3 '''

path = '/Lab1/Lab3/MRI/'
img_h, img_w = 240, 240
Mask = sorted(gen_list(path, 'Mask'))
Img = sorted(gen_list(path,'Image'))

Mask_train, Mask_val, Img_train, Img_val = shuffle_split(Mask, Img, 0.8)
Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w)
Mask_val = read_data(path+'Mask/', Mask_val, img_h, img_w)
Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
Img_val = read_data(path+'Image/', Img_val, img_h, img_w)

data_gen_args = dict(rotation_range=5,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     validation_split=0.2)
image_train_datagen = ImageDataGenerator(**data_gen_args)
mask_train_datagen = ImageDataGenerator(**data_gen_args)
image_val_datagen = ImageDataGenerator(**data_gen_args)
mask_val_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
batch_size=8

model = get_UNet(img_shape=(img_h, img_w, 1), Base=16, depth=4, inc_rate=2, 
                 activation='relu', drop=0, batchnorm=True, N=2)

model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], 
              metrics=[dice_coef,recall,precision])

# Train model
History = model.fit_generator(XYaugmentGenerator(image_train_datagen, mask_train_datagen, Img_train, Mask_train, seed, batch_size), 
                    steps_per_epoch=np.ceil(float(len(Img_train)) / float(batch_size)), 
                    validation_data = XYaugmentGenerator(image_val_datagen, mask_val_datagen, Img_val, Mask_val,seed, batch_size), 
                    validation_steps = np.ceil(float(len(Img_val)) / float(batch_size)), 
                    shuffle=True, epochs=150)

plot_learning_curve(History, 'Task3_loss')
plot_validation_metric_3(History, 'Task3_metrics')