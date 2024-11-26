# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:27:06 2024

@author: Annika
"""
import torch
import torch.nn as nn 
import torch.nn.functional as f
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow


#Load the data
torch.manual_seed(41)
dataset_dir="/Users/Annika/Documents/Year3/DeepLearning/Challenge/dl2425_challenge_dataset"
print(os.listdir(dataset_dir))

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to the same size
    transforms.ToTensor()          # Convert images to PyTorch tensors
])

# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "train").replace("\\", "/"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "val").replace("\\", "/"), transform=transform)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Class to Index Mapping:", train_dataset.class_to_idx)


#Data augmentation

#split in train and test
def extract_data(loader):
    images = []
    labels = []
    for data, label in loader:
        images.append(data)
        labels.append(label)
    # Stack the data and labels into a single tensor
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return images, labels

# Extract X_train, Y_train, X_val, Y_val
x_train, y_train = extract_data(train_loader)
x_val, y_val = extract_data(val_loader)


#Model: create Sequential model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(50, 50, 3)))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation="relu")) #second conv layer
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation="relu")) #third conv layer
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))


model.summary()

#compile model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=5)

#Try training the model for 30 epochs:

history=model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=30,callbacks=[early_stop],shuffle=True)