#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import random
from random import shuffle
from skimage.io import imread,imshow
from skimage.transform import resize
from itertools import chain
import cv2

# Data Loader                                
def gen_list(path, directory):
    
    list_ = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if directory in r:
                list_.append(os.path.join(r, file))
    return list_

# Shuffle
def shuffle(list1, list2):
    
    combination = list(zip(list1, list2))
    random.shuffle(combination)
    list1, list2 = zip(*combination)
    
    return list1, list2

#Split for K-flod cross validation
def split_list(list_,k):
    list_store = []
    num = int(len(list_)/k)
    num_ = num+1
    remainder = len(list_)%k
    mid_k = k - remainder
    mid_ind = mid_k*num
    for j in range(k):
        if j < mid_k:
            list_store.append(list_[j*num : (j+1)*num])
        else:
            list_store.append(list_[mid_ind + (j-mid_k)*num_ : 
                                    mid_ind + (j-mid_k+1)*num_]) 
    
    return list_store

def get_len(inp):
    return len(list(chain.from_iterable(inp)))


# Read data
def read_data(data_path, list_, img_h, img_w, binary=False):
    data = []       
    for i in range(len(list_)):
        image_name = list_[i]
        img = imread(os.path.join(data_path, image_name), as_gray=True)
        #img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        img = resize(img, (img_h, img_w), order=0, anti_aliasing=False, 
                     preserve_range=True).astype('float32')

        data.append(np.array(img))
        
    data = np.expand_dims(data, axis = 3)    
    print('Reading finishied')
    if binary:
        return data/np.max(data)
    else:
        return data

def binary_mask(mask, radius):
    bi_Mask = []
    for i in range(len(mask)):
        kernel = np.ones((radius, radius), np.uint8)
        dilation = cv2.dilate(mask[i], kernel)
        erosion = cv2.erode(dilation, kernel)
        boundary = cv2.subtract(dilation, erosion)
        boundary = np.expand_dims(boundary, axis=2)
        boundary = (boundary)/float(np.max(boundary))
        bi_Mask.append(boundary)
    bi_Mask = np.array(bi_Mask)
    
    return bi_Mask

