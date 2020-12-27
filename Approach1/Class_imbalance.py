#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:09:27 2020

@author: caroline
"""
import numpy as np
#import torchvision
import torch
#import matplotlib.pyplot as plt
from LoadData_ny import Data_train
from LoadData_ny import Data_val
from LoadData_ny import Data_test
from LoadData_ny import Classes
from LoadData_ny import categories
#from LoadData_ny import categories_index
#from LoadData_ny import categories_examples
#from LoadData_ny import show

# Calculating number of images in each class of train, val and test data 
k_train = np.zeros((Classes))
for i in range(len(Data_train)):
    # train
    (im, tar) = Data_train.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    for k in range(len(j)):
        k_train[j[k]] += 1
   
k_val = np.zeros((Classes))
for i in range(len(Data_val)):
    # val 
    (im, tar) = Data_val.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    for k in range(len(j)):
        k_val[j[k]] += 1
    
k_test = np.zeros((Classes))
for i in range(len(Data_test)):
    # test 
    (im, tar) = Data_test.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    for k in range(len(j)):
        k_test[j[k]] += 1

## Find most underrepresented class in train 
categories[np.argmin(k_train)] # boiled peas
minimum = min(k_train) # 103

class_count = np.zeros((Classes))
subset_array = []
for i in range(len(Data_train)):
    (im, tar) = Data_train.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    added = 0
    for k in range(len(j)):
        if class_count[j[k]] < minimum:
            added += 1
    if added == len(j):
        subset_array.append(i)
        for k in range(len(j)):   
            class_count[j[k]] += 1
        
Data_train_modified = torch.utils.data.Subset(Data_train, subset_array)

# Calculating number of images in each class of train, val and test data 
k_train = np.zeros((Classes))
for i in range(len(Data_train_modified)):
    # train
    (im, tar) = Data_train_modified.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    for k in range(len(j)):
        k_train[j[k]] += 1

## Find most underrepresented class in val
categories[np.argmin(k_val)] 
minimum = min(k_val) 

class_count = np.zeros((Classes))
subset_array = []
for i in range(len(Data_val)):
    (im, tar) = Data_val.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    added = 0
    for k in range(len(j)):
        if class_count[j[k]] < minimum:
            added += 1
    if added == len(j):
        subset_array.append(i)
        for k in range(len(j)):   
            class_count[j[k]] += 1
        
Data_val_modified = torch.utils.data.Subset(Data_val, subset_array)

# Calculating number of images in each class of train, val and test data 
k_val = np.zeros((Classes))
for i in range(len(Data_val_modified)):
    # train
    (im, tar) = Data_val_modified.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    for k in range(len(j)):
        k_val[j[k]] += 1

## Find most underrepresented class in test
categories[np.argmin(k_test)] 
minimum = min(k_test) 

class_count = np.zeros((Classes))
subset_array = []
for i in range(len(Data_test)):
    (im, tar) = Data_test.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    added = 0
    for k in range(len(j)):
        if class_count[j[k]] < minimum:
            added += 1
    if added == len(j):
        subset_array.append(i)
        for k in range(len(j)):   
            class_count[j[k]] += 1
        
Data_test_modified = torch.utils.data.Subset(Data_test, subset_array)

# Calculating number of images in each class of train, val and test data 
k_test = np.zeros((Classes))
for i in range(len(Data_test_modified)):
    # train
    (im, tar) = Data_test_modified.__getitem__(i)
    ntar = tar.numpy()
    j = np.where(ntar == 1)[0]
    for k in range(len(j)):
        k_test[j[k]] += 1


# ----------- function to flip images ---------------------------------------
#flip = torchvision.transforms.RandomHorizontalFlip(p=1)

#classes_upsample = [0,7,8] # peas, rice, spaghetti 
#im_list = []
#im_tar = []
#for i in range(len(Data_train)):
 #   (im, tar) = Data_train.__getitem__(i)
  #  ntar = tar.numpy()
  #  j = np.where(ntar == 1)[0][0]
   # if j in classes_upsample:
    #    im_flip = flip(im)
     #   im_list.append(im_flip)
      #  im_tar.append(tar)

# Plot examples 
#(rice, tar) = Data_train.__getitem__(categories_index[2]) #rice
#(spaghetti, tar) = Data_train.__getitem__(categories_index[4]) #spaghetti 
#(peas, tar) = Data_train.__getitem__(categories_index[9]) #peas

#show(rice)
#show(im_list[0])
#show(spaghetti)
#show(im_list[1])
#show(peas)
#show(im_list[5])







