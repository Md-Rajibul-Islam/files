# Task3
from dataloader import gen_list, shuffle_split, read_data
from Unet import get_UNet
from plot import plot_learning_curve
from metrics import dice_coef_loss, dice_coef

# Read the data
path = '/Lab1/Lab3/X_ray/'
img_h, img_w = 256, 256
Mask = gen_list(path, 'Mask')
Img = gen_list(path,'Image')


Mask_train, Mask_val, Img_train, Img_val = shuffle_split(Mask, Img, 0.8)
Mask_train = read_data(path+'Mask/', Mask_train, img_h, img_w)
Mask_val = read_data(path+'Mask/', Mask_val, img_h, img_w)
Img_train = read_data(path+'Image/', Img_train, img_h, img_w)
Img_val = read_data(path+'Image/', Img_val, img_h, img_w)

# Train the model
model = get_UNet(img_shape=(256,256,1), Base=32, depth=4, inc_rate=2, 
                 activation='relu', drop=0.5, batchnorm=False)


model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], 
              metrics=[dice_coef])

History = model.fit(Img_train, Mask_train, batch_size=8, epochs=150, verbose=2, 
                    validation_data=(Img_val, Mask_val))

# Plot the learning curve  
plot_learning_curve(History, 'Task3')
