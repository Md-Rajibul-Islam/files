#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import random
from random import shuffle
from skimage.io import imread,imshow
from skimage.transform import resize


# Data Loader                                
def gen_list(path, directory):
    
    list_ = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if directory in r:
                list_.append(os.path.join(r, file))
    return list_
                

# Shuffle and split
def shuffle_split(list1, list2, split_percent):
    
    combination = list(zip(list1, list2))
    random.shuffle(combination)
    list1, list2 = zip(*combination)
    list1_train = list1[:int(split_percent*len(list1))]
    list1_val = list1[int(split_percent*len(list1)):]
    list2_train = list2[:int(split_percent*len(list1))]
    list2_val = list2[int(split_percent*len(list1)):]
    
    return list1_train, list1_val, list2_train, list2_val

# Read data
def read_data(data_path, list_, img_h, img_w):
    data = []       
    for i in range(len(list_)):
        image_name = list_[i]
        img = imread(os.path.join(data_path, image_name), as_gray=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        #img = resize(img, (img_h, img_w), order=0, anti_aliasing=False, 
         #            preserve_range=True).astype('float32')

        data.append(np.array(img))
        
    data = np.expand_dims(data, axis = 3)    
    print('Reading finishied')         
    return data


def read_mask_onehot(data_path, list_, img_h, img_w):
    data = []       
    for i in range(len(list_)):
        image_name = list_[i]
        img = imread(os.path.join(data_path, image_name), as_gray=True)
        #img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        img = resize(img, (img_h, img_w), order=0, anti_aliasing=False, 
                     preserve_range=True).astype('float32')

        data.append(np.array(img))
        
    # One-hot encoding
    M = []
    for m in np.unique(data):
        M.append(data==m)
    data = np.transpose(np.asarray(M), (1,2,3,0))    
    
    print('Reading finishied')         
    return data