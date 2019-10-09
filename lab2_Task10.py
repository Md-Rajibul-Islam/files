# Task 10
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D
from tensorflow.keras.layers import Convolution2D 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications
import numpy as np
import matplotlib.pyplot as plt
from plot_lab2 import plot_learning_curve

def model_VGG(img_width, img_height, img_ch):

    model = Sequential()
    model.add(Conv2D(64, input_shape=(img_width, img_height, img_ch), kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', 
                     name = 'Last_ConvLayer'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))
       
    return model


# In[4]:


img_width, img_height, img_ch= 224, 224, 3
model = model_VGG(img_width, img_height, img_ch)

train_data_dir = '/Lab1/Lab2/Bone/train/'
validation_data_dir = '/Lab1/Lab2/Bone/validation/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1, 
    height_shift_range=0.1,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=8,
        class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=8,
        class_mode='binary')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer = Adam(lr=0.00001),
              metrics=['accuracy'])

History = model.fit_generator(train_generator,epochs=28,
                              verbose=1,validation_data=validation_generator)
plot_learning_curve(History, 'Task10')


# In[29]:


from tensorflow.keras import backend as K
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import cv2
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

Sample = '/Lab1/Lab2/Bone/train/AFF/14.jpg'
Img = imread(Sample)
#Img = Img[:,:,0]
Img = Img/255
Img = resize(Img, (img_height, img_width), anti_aliasing = True).astype('float32')
#Img = np.expand_dims(Img, axis = 2) 
Img = np.expand_dims(Img, axis = 0)
preds = model.predict(Img)
class_idx = np.argmax(preds[0])
print(class_idx)
class_output = model.output[:, class_idx]
last_conv_layer = model.get_layer("Last_ConvLayer")

grads = K.Gradient(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([Img])
for i in range(Base*8):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# For visualization
img = cv2.imread(Sample)
img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(superimposed_img)
