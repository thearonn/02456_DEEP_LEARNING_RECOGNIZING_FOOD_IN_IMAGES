# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:11:38 2020

@author: lisbe
"""

import torch.nn as nn
import torch
import torch.optim as optim
#import numpy as np
from LoadData_5dec import Classes
from LoadData_5dec import size
from LoadData_5dec import k_train

## ---------- Define hyperparameters --------------------------------
In_channels=3
Filters=16
KernelSize=7
Stride=1
Padding=0
Pool_dim=2
Pool_stride=2
dropout_rate=0.3
# Collect parametes in one array 
Parameters=[Classes, size, In_channels, Filters, KernelSize, Stride, Padding, Pool_dim, Pool_stride, dropout_rate]

## --------- Define functions to be used in the network --------------
def Conv_seq(In_channels, Filters, KernelSize, Stride, Padding,Pool_dim, Pool_stride, dropout_rate):
    return nn.Sequential(
          nn.Conv2d(In_channels, Filters, KernelSize, Stride, Padding),
          nn.ReLU(),
          nn.MaxPool2d(Pool_dim,Pool_stride), 
          nn.BatchNorm2d(Filters),
          nn.Dropout(p=dropout_rate)
          )

def compute_conv_dim(dim_size, KernelSize, Padding, Stride):
    return torch.ceil(torch.tensor((dim_size - KernelSize + 2 * Padding)/(Stride + 1))).item()  

def Lin_seq(in_features, dropout_rate):
    return nn.Sequential(
              nn.Linear(in_features, 120),
              nn.Linear(120, 84),
              nn.BatchNorm1d(84),
              nn.ReLU(inplace=True),
              nn.Dropout(p=dropout_rate))

def Last_layer(in_features, Classes):
    return nn.Linear(in_features,Classes)

## ------------ Define network ----------------------------------------
class Net(nn.Module):
    def __init__(self, parameters):
        super(Net, self).__init__()
        # define values
        Classes=parameters[0]
        dim=parameters[1] #start with image size
        In_channels=parameters[2]
        Filters=parameters[3]
        KernelSize=parameters[4]
        Stride=parameters[5]
        Padding=parameters[6]
        Pool_dim=parameters[7]
        Pool_stride=parameters[8]
        dropout_rate=parameters[9]
        self.FromConvToLin=0
        
        self.FirstConv = Conv_seq(In_channels, Filters, KernelSize, Stride, Padding, Pool_dim, Pool_stride, dropout_rate)
        dim = compute_conv_dim(dim, KernelSize, Padding, Stride)
        
        In_channels = Filters
        Out_channels = 20
        self.SecondConv = Conv_seq(In_channels, Out_channels, KernelSize, Stride, Padding, Pool_dim, Pool_stride, dropout_rate)
        dim = compute_conv_dim(dim, KernelSize, Padding, Stride)
        
        In_channels = Out_channels
        Out_channels = 15
        self.ThirdConv = Conv_seq(In_channels, Out_channels, KernelSize, Stride, Padding, Pool_dim, Pool_stride, dropout_rate)
        dim = compute_conv_dim(dim, KernelSize, Padding, Stride)
        
        In_features = int(Out_channels * dim * dim)
        
        self.FromConvToLin = In_features
        self.Linear = Lin_seq(In_features, dropout_rate)
        self.LastLayer=Last_layer(84, Classes)
        self.sig = nn.Sigmoid()
       
            
    def forward(self, x):
        x = self.FirstConv(x)
        x = self.SecondConv(x)
        x = self.ThirdConv(x)
        x = x.view(-1, self.FromConvToLin)
        x = self.Linear(x)
        x = self.LastLayer(x)
        x=self.sig(x) #udkommenter? lader ikke til atgøre en forskel
        return x

## ---------- Define Loss function and optimizer ------------------------
Net = Net(Parameters)
LR = 0.01
Momentum = 0.9
n_pos_train = k_train
n_neg_train = 703-n_pos_train
pos_w = n_neg_train/n_pos_train
#temp=1/(np.array(n_val_class)/nval_tot)
#criterion = nn.BCEWithLogitsLoss(pos_weight=2*torch.ones(10))
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(1*pos_w)) 
opt = optim.SGD(Net.parameters(), lr=LR, momentum=Momentum)


