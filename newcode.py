# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:44:33 2024

@author: Annika
"""

# %%
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt

print("Current working directory:", os.getcwd())

# %%

dataset_dir = "dl2425_challenge_dataset"
print(os.listdir(dataset_dir))


# preproccessing step, augmentation, to tensor object
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder(os.path.join(
    dataset_dir, "train").replace("\\", "/"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(
    dataset_dir, "val").replace("\\", "/"), transform=transform)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

print("Class to Index Mapping:", train_dataset.class_to_idx)

















# %%

#use a pretrained model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Since this is a binary classification task, we'll set the size of each output sample to 2. For multi-class classification, this can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# ptimize all parameters of the model
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-5)

# We'll decay learning rate (lr) by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("Resnet downloaded, set up")


# %%

# Model creation:
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
criterion_cnn = torch.nn.CrossEntropyLoss()  # Optimizer
optimizer_cnn = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler_cnn = lr_scheduler.StepLR(optimizer_cnn, step_size=7, gamma=0.1)















# %%


# valid_loss_min = np.inf # track change in validation loss

def train_model(model, train_loader, criterion, optimizer, scheduler, device):
    n_epochs= 2
    # Iterate over epochs
    for epoch in range(1, n_epochs + 1):

        # Keep track of training loss
        train_loss = 0.0

        # Set model to training mode
        model.train()

        # Iterate over batches in the training set
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate the batch loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            # Update training loss
            train_loss += loss.item() * data.size(0)

            scheduler.step()

        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)


        # Print the training loss for the current epoch
        print(f"Epoch {epoch}, Average Training Loss: {train_loss:.6f}")

    # Return the trained model
    return model


# %%
def evaluate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0   # Count of correct predictions
    total = 0     # Total samples in the test set

    with torch.no_grad():  # Disable gradient computation for evaluation
        for i, (data, target) in enumerate(val_loader):
            # Get model predictions: forward pass
            output = model(data)

            # Predicted class
            _, predicted_class = torch.max(output, 1)

            # Count correct predictions
            correct += (predicted_class == target).sum().item()
            total += target.size(0)

            # Print individual predictions for debugging
            for j in range(len(data)):
                print(f"{i * len(data) + j + 1}.) Prediction: {predicted_class[j].item()} | "
                      f"Actual: {target[j].item()}")

    # Calculate and print overall accuracy
    accuracy = correct / total
    print(f"\nOverall Test Accuracy: {accuracy:.2%}")
    print(f"Total Correct Predictions: {correct}/{total}")



















# %%
#train with pretrained model
model_ft = train_model(model_ft, train_loader, criterion, optimizer_ft, exp_lr_scheduler, device)


# %%
#train CNN
model = train_model(model, train_loader, criterion_cnn, optimizer_cnn, exp_lr_scheduler_cnn, device)





# %%
#evaluating model 
evaluate_model(model_ft, val_loader)









# %%
# Save the model
torch.save(model_ft.state_dict(), 'model_fire_pretrained.pt')
torch.save(model.state_dict(), 'model_fire.pt')
# %%

# Testing unlabelled images
# Create csv file with img id and predicted class
# transform the test dataset the same way we have transformed train
csv_file_path = "test_data.csv"
image_folder = "dl2425_challenge_dataset/test"
with open(csv_file_path, mode='w', newline='') as file:
    fieldnames = ["img_id", "predicted_class"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header row
    with torch.no_grad():
        for f in os.listdir(image_folder):
            img_path = os.path.join(image_folder, f)
            img = Image.open(img_path).convert('RGB')
            image_tensor = transform(img)
            output = model(image_tensor)  # Get model predictions
            _, predicted_class = torch.max(output, 1)
            img_id = os.path.basename(f)  # Get the image filename
            writer.writerow(
                {"img_id": img_id, "predicted_class": predicted_class.item()})


# %%
# plot random image from csv file + prediction
random_index = random.randint(2, 1563)
img_id, predicted_class = prediction(image_folder, csv_file_path, random_index)


def prediction(image_folder, csv_file_path, index):
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    row = rows[index]
    img_id = row['img_id']
    img_id = row['img_id']
    predicted_class = int(row['prediction'])
    img_path = os.path.join(image_folder, img_id)
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    plt.title(f"Prediction: Class {predicted_class}")
    plt.axis('off')
    plt.show()

    return img_id, predicted_class


# %%
"""        
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
    # save model if validation loss has decreased
# save model if validation loss has decreased
if valid_loss <= valid_loss_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
    torch.save(model.state_dict(), 'model_cifar.pt')
    valid_loss_min = valid_loss
"""
