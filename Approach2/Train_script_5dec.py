#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:19:37 2020

@author: caroline
"""

import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
from LoadData_5dec import Data_train
from LoadData_5dec import Data_val
from LoadData_5dec import Data_test
from LoadData_5dec import categories, Classes
from NetAdvanced_ny_5dec import Net 
from NetAdvanced_ny_5dec import opt
from NetAdvanced_ny_5dec import criterion

# define model name
Modelname = 'Net.Advanced.pt'

# ----------- Define parameters for training --------------------------------
batch_size = 15
num_epoch = 50+1
train_acc, train_loss = [], []
val_acc, val_loss = [], []
losses = []
lowest_loss = 1000
epoch_array=[]


#--------------------------------------------------------------
train_loader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size,shuffle=True,drop_last=True)
val_loader = torch.utils.data.DataLoader(Data_val,batch_size=batch_size,shuffle=True,drop_last=True)

## ----------- Train and validation loop ------------------------------------

for epoch in range(num_epoch):
    
    ## TRAIN
    print('Training: Starting epoch {}/{}'.format(epoch+1, num_epoch))
    running_train_loss = 0.0
    pbar = enumerate(train_loader) # len(list(enumerate(train_loader))) = len(Data_train)/batch_size
    count=0
    # Looping through every image in the batch 
    for i, (image, target) in pbar:
        images = image
        print('batch size number: ', count+1)
        count+=1
        target=target
    
        Net.train() 
        opt.zero_grad()
        output = Net(images.float()) 
        loss = criterion(output, target.float())
        
        # Computes gradients
        loss.backward()   
        opt.step()
        
        running_train_loss += loss.item() 
        
        
    # append loss for each epoch
    num_samples = float(len(train_loader.dataset))
    train_loss.append(running_train_loss/num_samples)
    
    ## VALIDATION
    Net.eval()
    running_val_loss = 0
    pbar = enumerate(val_loader)
    
    print('Validation: Starting epoch {}/{}'.format(epoch+1, num_epoch))
    for i, (images,target) in pbar:
        images=image
        target=target
        
        output = Net(images.float()) 
        
        loss_val = criterion(output, target.float())
        
        running_val_loss += loss_val.item()
    
    # append epoch number to array 
    epoch_array.append(epoch)
    # append loss for each epoch
    num_samples = float(len(val_loader.dataset))
    val_loss.append(running_val_loss/num_samples)

    # Save weights of best network
    if running_val_loss < lowest_loss:
        lowest_loss = running_val_loss
        best_model = copy.deepcopy(Net.state_dict())

# Save model
torch.save(best_model, Modelname)

# Plot training and validation
plt.plot(epoch_array,train_loss,epoch_array,val_loss)
plt.ylim(0.05,0.085)
plt.legend(["Training data", "Validation data"])
title = 'Batch size ' + str(batch_size) + ' epochs ' + str(epoch)
plt.title(title)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show


# ------------------------Testing------------------------------------------------------------
from torch.autograd import Variable

test_loader = torch.utils.data.DataLoader(Data_test, batch_size=batch_size,shuffle=True,drop_last=True)
test_enum = list(enumerate(test_loader))

conf_mat=[]
recall=[]
precision=[]
f1_score=[]
acc=[]
        
for i_class in range(Classes):
    
    #true positives
    TP = np.zeros((Classes))
    TN = np.zeros((Classes))
    FP = np.zeros((Classes))
    FN = np.zeros((Classes))
        
    for i_batch in range(len(test_enum)):
        test_image = test_enum[i_batch][1][0]
        test_target = test_enum[i_batch][1][1]
        
        outputs = Net(Variable(test_image))
        for i in range(batch_size):
            prob_class = outputs.data[:][i]
            
            pred_class = np.array(1*(prob_class > 0.5))
            
            #for i in range(len(prob_class)):
             #   if (prob_class[i]>0.5):
              #      pred_class=1
               # else:
                #    pred_class=0
            
            
            for j in range(len(pred_class)):
                label = test_target[i][j]
                if(pred_class[j] == 1 and label == 1):
                    TP[j] += 1
                elif(pred_class[j] == 1 and label == 0):
                    FP[j] += 1
                elif(pred_class[j] == 0 and label == 1):
                    FN[j] += 1
                else:
                    TN[j] += 1
          
    conf_mat = np.array([[TP[i_class], FP[i_class]],[FN[i_class],TN[i_class]]])
    print('confusion matrix for category', categories[i_class])
    print(conf_mat)
    if(TP[i_class] + FN[i_class] == 0):
        recall_class=0
    else:
        recall_class = TP[i_class]/(TP[i_class]+FN[i_class])
    #recall.append(recall_class)
    print('recall for category', categories[i_class])
    print(recall_class)
    if(TP[i_class]+FP[i_class] == 0):
        precision_class = 0
    else:
        precision_class = TP[i_class]/(TP[i_class]+FP[i_class])
    #precision.append(precision_class)
    print('precision for category', categories[i_class])
    print(precision_class)
    if(recall_class+precision_class == 0):
        f1_class = 0
    else: 
        f1_class = 2*recall_class*precision_class/(recall_class+precision_class)
    #f1_score.append(f1_class)
    print('f1 score for class', categories[i_class])
    print(f1_class)
    acc_class = (TN[i_class]+TP[i_class])/(TN[i_class]+TP[i_class]+FN[i_class]+FP[i_class])
    #acc.append(acc_class)
    print('accuracy for class', categories[i_class])
    print(acc_class)
    print('')
    
# Total confusion matrix
conf_mat_total = np.array([[sum(TP), sum(FP)],[sum(FN), sum(TN)]])
print('confusion matrix total')
print(conf_mat_total)
if(sum(TP) + sum(FN) == 0):
    recall_total=0
else:
    recall_total = sum(TP)/(sum(TP)+sum(FN))
print('recall total')
print(recall_total)
if(sum(TP)+sum(FP) == 0):
    precision_total = 0
else:
    precision_total = sum(TP)/(sum(TP)+sum(FP))
print('precision total')
print(precision_total)
if(recall_total+precision_total == 0):
    f1_total = 0
else: 
    f1_total = 2*recall_total*precision_total/(recall_total+precision_total)
print('f1 score total')
print(f1_total)
acc_total = (sum(TN)+sum(TP))/(sum(TN)+sum(TP)+sum(FN)+sum(FP))
print('accuracy total')
print(acc_total)
print('')




