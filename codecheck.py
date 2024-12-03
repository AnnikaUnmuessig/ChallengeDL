# %%
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt

# %% Load Dataset

dataset_dir = "dl2425_challenge_dataset"
print(os.listdir(dataset_dir))

# Preprocessing steps: augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder(
    os.path.join(dataset_dir, "train").replace("\\", "/"), transform=transform)
val_dataset = datasets.ImageFolder(
    os.path.join(dataset_dir, "val").replace("\\", "/"), transform=transform)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

print("Class to Index Mapping:", train_dataset.class_to_idx)

# %% Define the Model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(0.3)

        # Dynamically calculate the flattened size
        # Input size: (1, 3, 224, 224)
        dummy_input = torch.zeros(1, 3, 224, 224)
        flattened_size = self._get_flattened_size(dummy_input)

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)

    def _get_flattened_size(self, x):
        # Pass dummy input through conv and pooling layers to compute size
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.view(-1).shape[0]  # Flatten and calculate size

    def forward(self, x):
        # Forward pass through conv, pooling, and fully connected layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Initialize the model
model = Net()

# %% Loss Function, Optimizer, and Scheduler

criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# %% Training Loop

n_epochs = 35  # Number of epochs

for epoch in range(1, n_epochs + 1):
    # Training phase
    model.train()
    train_loss = 0.0

    for data, target in train_loader:
        optimizer.zero_grad()  # Clear gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        train_loss += loss.item() * data.size(0)  # Accumulate training loss

    train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}, Average Training Loss: {train_loss:.6f}")

# %% Validation Phase

model.eval()  # Set the model to evaluation mode
correct = 0   # Count of correct predictions
total = 0     # Total samples in the validation set

with torch.no_grad():  # Disable gradient computation
    for data, target in val_loader:
        output = model(data)  # Get model predictions
        _, predicted_class = torch.max(output, 1)  # Predicted class
        correct += (predicted_class == target).sum().item()  # Count correct
        total += target.size(0)

accuracy = correct / total  # Overall accuracy
print(f"\nValidation Accuracy: {accuracy:.2%} ({correct}/{total})")

# %% Save the Model

torch.save(model.state_dict(), 'model_fire.pt')

# %% Test on Unlabeled Images

csv_file_path = "test_data.csv"
image_folder = "dl2425_challenge_dataset/test"

with open(csv_file_path, mode='w', newline='') as file:
    fieldnames = ["img_id", "predicted_class"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    with torch.no_grad():
        for f in os.listdir(image_folder):
            img_path = os.path.join(image_folder, f)
            img = Image.open(img_path).convert('RGB')
            image_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            output = model(image_tensor)  # Get model predictions
            _, predicted_class = torch.max(output, 1)
            img_id = os.path.basename(f)
            writer.writerow(
                {"img_id": img_id, "predicted_class": predicted_class.item()})

# %% Visualize a Random Prediction


def prediction(image_folder, csv_file_path, index):
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    row = rows[index]
    img_id = row['img_id']
    predicted_class = int(row['predicted_class'])
    img_path = os.path.join(image_folder, img_id)
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    plt.title(f"Prediction: Class {predicted_class}")
    plt.axis('off')
    plt.show()

    return img_id, predicted_class


random_index = random.randint(0, len(os.listdir(image_folder)) - 1)
img_id, predicted_class = prediction(image_folder, csv_file_path, random_index)
