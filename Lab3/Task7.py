# Task7
from dataloader import gen_list, shuffle_split, read_data
from Unet import get_UNet
from plot import plot_learning_curve, plot_validation_metric
from metrics import dice_coef_loss, dice_coef, precision, recall

# Read the data
path = '/Lab1/Lab3/MRI/'
img_h, img_w = 240, 240
Mask = gen_list(path, 'Mask')
Img = gen_list(path,'Image')

Mask_train, Mask_val, Img_train, Img_val = shuffle_split(Mask, Img, 0.8)
Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w)
Mask_val =read_data(path+'Mask/', Mask_val, img_h, img_w)
Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
Img_val = read_data(path+'Image/', Img_val, img_h, img_w)

# Train the model
model = get_UNet(img_shape=(240,240,1), Base=32, depth=4, inc_rate=2, 
                 activation='relu', drop=0.5, batchnorm=True)

model.compile(optimizer=Adam(lr=0.00001), loss=[dice_coef_loss], 
              metrics=[dice_coef,precision,recall])

History = model.fit(Img_train, Mask_train, batch_size=4, epochs=50, verbose=1, 
                    validation_data=(Img_val, Mask_val))

# Plot learning curves
plot_learning_curve(History, 'Task7_1')
plot_validation_metric(History, 'Task7_2')
