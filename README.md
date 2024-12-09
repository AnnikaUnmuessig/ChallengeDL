# 🧠 Binary Classification using ResNet-50

This project implements a **binary classification model** using PyTorch. The goal is to train a deep learning model to classify input data into one of two categories. 🚀 The project also includes real-time visualization of training progress through loss and accuracy plots 📊.

---

## 📋 Table of Contents

- [📖 Overview](#-overview)
- [✨ Features](#-features)
- [⚙️ Prerequisites](#️-prerequisites)
- [📂 Dataset](#-dataset)
- [📦 Installation](#-installation)
- [📈 Results](#-results)

---

## 📖 Overview

Binary classification is a common machine learning task where the goal is to assign one of two labels to input data. This project demonstrates the following:

1. 🏋️‍♂️ Training a binary classifier using PyTorch.
2. 📏 Calculating and tracking metrics like loss and accuracy.
3. 📊 Visualizing training and validation performance over epochs.

The training pipeline includes:
- 🗂 Data pre-processing and loading.
- 🔄 Model training and evaluation.
- 📉 Learning rate scheduling.
- 🖼 Metric visualization.

---

## ✨ Features

- ✅ **Training and Validation Loop**: Implements separate phases for training and validation.
- 🧪 **Binary Classification**: Uses sigmoid activation to handle binary outputs.
- 📊 **Visualization**: Generates plots for training and validation loss and accuracy.
- 💾 **Checkpoints**: Saves the best model based on validation accuracy.

---

## ⚙️ Prerequisites
- 📋 Check the [requirements](requirements.txt)

---

## 📂 Dataset

- **`data/`**
  - **`train/`** (Training data)
    - **`class_0/`** (Images for class 0)
      - `img1.jpg`
      - `img2.jpg`
      - ...
    - **`class_1/`** (Images for class 1)
      - `img3.jpg`
      - `img4.jpg`
      - ...
  - **`val/`** (Validation data)
    - **`class_0/`** (Images for class 0)
      - `img5.jpg`
      - `img6.jpg`
      - ...
    - **`class_1/`** (Images for class 1)
      - `img7.jpg`
      - `img8.jpg`
      - ...
  - **`test/`** (Test data)
      - `img10.jpg`
      - `img11.jpg`
      - ...


## 🔧 Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/AnnikaUnmuessig/ChallengeDL.git
   cd ChallengeDL
   ```
2. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset is properly structured (see Dataset).

---

## 📈 Results

### 🏆 Best Model Performance
- **Validation Accuracy**: `98.78%` 
- **Training Loss Trends**: Training loss consistently decreased over epochs.
- **Validation Loss Trends**: Validation loss stabilized, showing minimal overfitting.

### 📊 Sample Visualizations
Here are the plots for training and validation loss and accuracy:

#### Loss Plot  
![Loss Plot](result/loss_plot.png)  

#### Accuracy Plot  
![Accuracy Plot](result/accuracy_plot.png)  

These visualizations help to understand the learning behavior of the model over the training process.

        



