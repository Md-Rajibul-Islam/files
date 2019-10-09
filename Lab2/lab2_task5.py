# Task 5b (2nd part)
import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)

import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

Sample = '/Lab1/X_ray/train/C4_4662.jpg'
Img = imread(Sample)
row, col = Img.shape

def show_paired(Original, Transform, Operation):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(Original, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(Transform, cmap='gray')
    ax[1].set_title(Operation + " image")
    if Operation == "Rescaled":
        ax[0].set_xlim(0, col)
        ax[0].set_ylim(row, 0)
    else:
        ax[0].axis('off')
        ax[1].axis('off')
    plt.tight_layout()

# Scaling
scale_factor = 0.2
image_rescaled = rescale(Img, scale_factor)
show_paired(Img, image_rescaled, "Rescaled")

# Roation
Angle = 30
image_rotated = rotate(Img, Angle)
show_paired(Img, image_rotated, "Rotated")

# Horizontal Flip
horizontal_flip = Img[:, ::-1]
show_paired(Img, horizontal_flip, 'Horizontal Flip')

# Vertical Flip
vertical_flip = Img[::-1, :]
show_paired(Img, vertical_flip, 'vertical Flip')


# Intensity rescaling
Min_Per, Max_Per = 5, 95
min_val, max_val = np.percentile(Img, (Min_Per, Max_Per))

better_contrast = exposure.rescale_intensity(Img, in_range=(min_val, max_val))
show_paired(Img, better_contrast, 'Intensity Rescaling')
# find the printing option
#plt.figure()
#plt.imshow(show_paired)


# Task 5b
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
#%matplotlib inline
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

Sample = '/Lab1/X_ray/train/C4_4662.jpg'
Img = imread(Sample)
Img = np.expand_dims(Img, axis = 2)
Img = np.expand_dims(Img, axis = 0)


count = 5
MyGen = ImageDataGenerator(rotation_range = 20,
                         width_shift_range = 0.2,
                         horizontal_flip = True)


fix, ax = plt.subplots(1,count+1, figsize=(14,2))
images_flow = MyGen.flow(Img, batch_size=1)
for i, new_images in enumerate(images_flow):
    new_image = array_to_img(new_images[0], scale=True)
    ax[i].imshow(new_image,cmap="gray")
    if i >= count:
        break



# Task 5b (2nd part)
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
#%matplotlib inline
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

Sample = '/Lab1/X_ray/train/C4_4662.jpg'
Img = imread(Sample)
Img = np.expand_dims(Img, axis = 2)
Img = np.expand_dims(Img, axis = 0)


count = 5
MyGen = aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
#total = 0


fix, ax = plt.subplots(1,count+1, figsize=(14,2))
images_flow = MyGen.flow(Img, batch_size=1)
for i, new_images in enumerate(images_flow):
    new_image = array_to_img(new_images[0], scale=True)
    ax[i].imshow(new_image,cmap="gray")
    if i >= count:
        break