# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:19:29 2020

@author: lisbe
"""

#import torch

import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


#define path for data
#path='\\Users\lisbe\Desktop\VOC'
path = '/Users/caroline/Documents/DeepLearning/Project/VOC/';

# define parameters for network
Classes=10
size=300 # image dimension (size * size)

# Function for reshaping images to dimension (size * size)
# Consider normalization: transforms.Normalize(mean = mean, std = std)?
transformations = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), ])


#--------------------------Define the categories-----------------
categories = ['boiled peas','boiled potatoes', 'chopped lettuce', 'fried egg', 
                     'glass of milk','glass of water','meatballs', 'plain rice',
                     'plain spaghetti', 'slice of bread']
C=len(categories)


# ONE HOT ENCODING function for the classes 
def encode_labels(target):
    #save the information about the target 
    info = target['annotation']['object']

    array=np.zeros(len(categories)) # array for the encoding
    c=[]# array to save the class in
    
    if type(info) == dict:
        if int(info['difficult']) == 0: #difficult: if it is not evaluated difficult is 1 else 0.
            c.append(categories.index(info['name']))
    else: 
        for i in range(len(info)):
            if int(info[i]['difficult']) == 0:
                c.append(categories.index(info[i]['name']))
    
    array[c] = 1
    #convert from numpy to tensor
    array=torch.from_numpy(array)
    
    return array

#-------------------------LOAD TRAINING DATA--------------------------
Data_train=torchvision.datasets.VOCDetection(root = path, year = '2007', image_set = 'train', download = False,
                                             transform=transformations, target_transform=encode_labels)

#alternative (load subset)
#subset_array = list(range(0, len(Data_train), 3))
#Data_train = torch.utils.data.Subset(Data_train, subset_array)

#-------------------------LOAD VALIDATION DATA--------------------------
Data_val=torchvision.datasets.VOCDetection(root = path, year = '2007', image_set = 'val', download = False,
                                             transform=transformations, target_transform=encode_labels)

#alternative (load subset)
#subset_array = list(range(0, len(Data_val), 3))
#Data_val = torch.utils.data.Subset(Data_val, subset_array)

#-------------------------LOAD TEST DATA--------------------------
Data_test=torchvision.datasets.VOCDetection(root = path, year = '2007', image_set = 'test', download = False,
                                            transform=transformations, target_transform=encode_labels)

#alternative (load subset)
#subset_array = list(range(0, int(len(Data_test)), 3))
#Data_test = torch.utils.data.Subset(Data_test, subset_array)

print('Images in training:', len(Data_train))
print('Images in validation:', len(Data_val))
print('Images in test:', len(Data_test))
print('Images in total: ', len(Data_train) + len(Data_val) + len(Data_test))
        

# ------------------ PLOT AN IMAGE FROM EACH CLASS IN TRAIN DATA ---------

# Function for ploting an image saved as a tensor
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

# Get index of one image from each category 
categories_examples = []
categories_index = []
for i in range(len(Data_train)):
    (im,tar) = Data_train.__getitem__(i)
    ntar = tar.numpy()
    k = np.where(ntar == 1)[0][0]
    if k not in categories_examples:
        categories_examples.append(k)
        categories_index.append(i)

# Plot images
for i in range(Classes):
    (im,tar) = Data_train.__getitem__(categories_index[i])
    plt.figure(figsize=(4, 4))
    show(im)
    plt.title("%s" % (categories[categories_examples[i]]))
    plt.axis('off')
    plt.show()   
 
















