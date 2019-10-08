import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize


img_w, img_h = 128, 128

# Assigning labels two images; those images contains pattern1 in their filenames
# will be labeled as class 0 and those with pattern2 will be labeled as class 1.

def gen_labels(im_name, pat1, pat2):
    if pat1 in im_name:
        Label = np.array([0])
    elif pat2 in im_name:
        Label = np.array([1])
    return Label


# reading and resizing the training images with their corresponding labels
def train_data(train_data_path, train_list):
    train_img = []
    for i in range(len(train_list)):
        image_name = train_list[i]
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing=True).astype('float32')
        train_img.append([np.array(img), gen_labels(image_name, 'Mel', 'Nev')])

        if i % 200 == 0:
            print('Reading: {0}/{1}  of train images'.format(i, len(train_list)))

    shuffle(train_img)
    return train_img


# reading and resizing the testing images with their corresponding labels
def test_data(test_data_path, test_list):
    test_img = []
    for i in range(len(test_list)):
        image_name = test_list[i]
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing=True).astype('float32')
        test_img.append([np.array(img), gen_labels(image_name, 'Mel', 'Nev')])

        if i % 100 == 0:
            print('Reading: {0}/{1} of test images'.format(i, len(test_list)))

    shuffle(test_img)
    return test_img


# Instantiating images and labels for the model.
def get_train_test_data(train_data_path, test_data_path, train_list, test_list):
    Train_data = train_data(train_data_path, train_list)
    Test_data = test_data(test_data_path, test_list)

    Train_Img = np.zeros((len(train_list), img_h, img_w), dtype=np.float32)
    Test_Img = np.zeros((len(test_list), img_h, img_w), dtype=np.float32)

    Train_Label = np.zeros((len(train_list)), dtype=np.int32)
    Test_Label = np.zeros((len(test_list)), dtype=np.int32)

    for i in range(len(train_list)):
        Train_Img[i] = Train_data[i][0]
        Train_Label[i] = Train_data[i][1]

    Train_Img = np.expand_dims(Train_Img, axis=3)

    for j in range(len(test_list)):
        Test_Img[j] = Test_data[j][0]
        Test_Label[j] = Test_data[j][1]

    Test_Img = np.expand_dims(Test_Img, axis=3)

    return Train_Img, Test_Img, Train_Label, Test_Label

