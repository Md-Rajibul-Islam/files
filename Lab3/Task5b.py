# Task5b
from dataloader import gen_list, shuffle_split, read_data
from Unet import get_UNet
from plot import plot_learning_curve, plot_validation_metric
from metrics import dice_coef_loss, dice_coef, precision, recall
from augmentation import XYaugmentGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Read the data
path = '/Lab1/Lab3/CT/'
img_h, img_w = 256, 256
Mask = gen_list(path, 'Mask')
Img = gen_list(path,'Image')

Mask_train, Mask_val, Img_train, Img_val = shuffle_split(Mask, Img, 0.8)
Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w)
Mask_val = read_data(path+'Mask/', Mask_val, img_h, img_w)
Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
Img_val = read_data(path+'Image/', Img_val, img_h, img_w)

# Data augmentation
data_gen_args = dict(rotation_range=5,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     validation_split=0.2)
image_train_datagen = ImageDataGenerator(**data_gen_args)
mask_train_datagen = ImageDataGenerator(**data_gen_args)
image_val_datagen = ImageDataGenerator(**data_gen_args)
mask_val_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
batch_size=4

# Train the model
model = get_UNet(img_shape=(256,256,1), Base=16, depth=4, inc_rate=2, 
                 activation='relu', drop=0.5, batchnorm=True)

model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], 
              metrics=[dice_coef, precision, recall])

History = model.fit_generator(XYaugmentGenerator(image_train_datagen, mask_train_datagen, Img_train, Mask_train, seed, batch_size), 
                    steps_per_epoch=np.ceil(float(len(Img_train)) / float(batch_size)), 
                    validation_data = XYaugmentGenerator(image_val_datagen, mask_val_datagen, Img_val, Mask_val,seed, batch_size), 
                    validation_steps = np.ceil(float(len(Img_val)) / float(batch_size)), 
                    shuffle=True, epochs=100)

# Plot the learning curve
plot_learning_curve(History, 'Task5b_1')
plot_validation_metric(History, 'Task5b_2')
