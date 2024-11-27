# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:44:33 2024

@author: Annika
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
import torchvision



dataset_dir="/Users/Annika/Documents/Year3/DeepLearning/Challenge/dl2425_challenge_dataset"
print(os.listdir(dataset_dir))



#preproccessing step, augmentation, to tensor object
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "train").replace("\\", "/"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "val").replace("\\", "/"), transform=transform)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

print("Class to Index Mapping:", train_dataset.class_to_idx)




#Model creation:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*53*53, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x
    
# create a complete CNN
model = Net()

# Loss function
criterion = torch.nn.CrossEntropyLoss()# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum= 0.9)


# number of epochs to train the model
n_epochs = 5

valid_loss_min = np.inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    # train the model #
    model.train()
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
        
        
        # Validating the model
model.eval()
for data, target in val_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update average validation loss 
    valid_loss += loss.item()*data.size(0)

# calculate average losses
train_loss = train_loss/len(train_loader.dataset)
valid_loss = valid_loss/len(val_loader.dataset)
    
# print training/validation statistics 
print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
    epoch, train_loss, valid_loss))