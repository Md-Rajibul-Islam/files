import pandas as pd
import numpy as np
from LSTM_model import lstm_model
from plot import plot_learning_curve, plot_validation_metric, plot_validation_metric_task2
from plot import plot_validation_metric_task3
from tensorflow.keras.optimizers import SGD, Adam

'''
#Task 1
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

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# split training data into T time steps:
X_train = []
y_train = []

T = 60
for i in range(T, len(training_set)):
    X_train.append(training_set_scaled[i - T:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# normalize the validation set according to the normalization applied to the training set:
inputs = dataset_total[len(dataset_total) - len(dataset_val) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# split validation data into T time steps:
X_val = []

for i in range(T, T + len(val_set)):
    X_val.append(inputs[i - T:i, 0])

X_val = np.array(X_val)
y_val = sc.transform(val_set)

# reshape to 3D array (format needed by LSTMs -> number of samples, timesteps, input dimension)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

units = 40
model = lstm_model(units, batch_size=16, input_size=X_train.shape[1], input_dimension=1)
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])
History = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1,
                    validation_data=(X_val, y_val))


plot_learning_curve(History, 'Task1_learning_curve_units={0}_loss'.format(units))
plot_validation_metric(History, 'Task1_validation_metric_units={0}_metrics'.format(units))
'''

'''
# Task 2
import nibabel as nib
from tensorflow.keras.utils import Sequence
dataPath = '/Lab1/Lab5/HCP_lab/'

def load_streamlines(dataPath, subject_ids, bundles, n_tracts_per_bundle):
    X = []
    y = []
    for i in range(len(subject_ids)):
        for c in range((len(bundles))):
            filename = dataPath + subject_ids[i] + '/' + bundles[c] + '.trk'
            tfile = nib.streamlines.load(filename)
            streamlines = tfile.streamlines

            n_tracts_total = len(streamlines)
            ix_tracts = np.random.choice(range(n_tracts_total), n_tracts_per_bundle, replace=False)
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


train_subjects_list = ['599469', '599671', '601127']  # your choice of 3 training subjects
val_subjects_list = ['613538']  # your choice of 1 validation subjects
bundles_list = ['CST_left', 'CST_right']
n_tracts_per_bundle = 20

X_train, y_train = load_streamlines(dataPath, train_subjects_list, bundles_list, n_tracts_per_bundle)
X_val, y_val = load_streamlines(dataPath, val_subjects_list, bundles_list, n_tracts_per_bundle)


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
        return int(np.floor(len(self.y) / self.batch_size))

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


units = 8
batch_size = 1
model = lstm_model(units, batch_size=batch_size, input_size=None, input_dimension=3)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
History = model.fit_generator(MyBatchGenerator(X_train, y_train, batch_size=1), epochs=50,
                              validation_data=MyBatchGenerator(X_val, y_val, batch_size=1),
                              validation_steps=len(X_val))

plot_learning_curve(History, 'Task2_learning_curve_units={0}_loss'.format(units))
plot_validation_metric_task2(History, 'Task2_validation_metric_units={0}_metrics'.format(units))
'''

# Task 3
from model_UNet import get_unet
from similarity_metrices import dice_coef_loss, dice_coef, precision, recall
from data_loader import shuffle_split, gen_list, read_data, read_mask_onehot_encoding

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


plot_learning_curve(History, 'Task3_learning_curve_with_encoding')
plot_validation_metric_task3(History, 'Task3_metrics_with_encoding')
print('Task 7 done !')