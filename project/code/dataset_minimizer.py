#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
from random import shuffle

def gen_list(path, directory):
    
    list_ = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if directory in r:
                list_.append(os.path.join(r, file))
    shuffle(list_)
    return list_

def collect(list_, path, class_, num_img):
    for i,name in enumerate(list_):
        if i<num_img:
            shutil.copy(name, path + class_ + '/')
    return()

path1 = '/dl_data/train/'
path2 = '/dl_data/validation/'

# Get the name list from whole dataset 
train_0 = gen_list(path1, '0')
train_1 = gen_list(path1, '1')
val_0 = gen_list(path2, '0')
val_1 = gen_list(path2,'1')

# Make new ddirectories
os.makedirs('/dl_data/tra/0/')
os.makedirs('/dl_data/tra/1/')
os.makedirs('/dl_data/val/0/')
os.makedirs('/dl_data/val/1/')
path_1 = '/dl_data/tra/'
path_2 = '/dl_data/val/'

# Move part of data to the new directories
collect(train_0, path_1, '0', 10000)
collect(train_1, path_1, '1', 5000)
collect(val_0, path_2, '0', 1500)
collect(val_1, path_2, '1', 750)

