from keras.preprocessing.image import ImageDataGenerator


aug = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True)
