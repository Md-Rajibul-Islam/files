from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Input, concatenate, BatchNormalization
from tensorflow.keras.models import Model

from keras.models import *
from keras.layers import *
import numpy as np

img_size = 240

def conv2d_block(input_tensor, n_filters, kernel_size, batchnorm=True):
    # first convolution layer
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second convolution layer
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def get_unet(input_img, n_filters, kernel_size, dropout, batchnorm=True):
    # Contracting Path
    input_img = Input(input_img)
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=kernel_size, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck block contains only the two convolution block
    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=kernel_size, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=2, padding='same')(c5)
    x1 = Reshape(target_shape=(1, np.int32(img_size / 8), np.int32(img_size / 8), n_filters * 8))(c4)
    x2 = Reshape(target_shape=(1, np.int32(img_size / 8), np.int32(img_size / 8), n_filters * 8))(u6)

    u6 = concatenate([x1, x2])
    u6 = ConvLSTM2D(n_filters * 4, (3, 3), padding='same', return_sequences=False, go_backwards=True)(u6)
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=True)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size=kernel_size, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    #model.summary()
    return model
