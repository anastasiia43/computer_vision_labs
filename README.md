# Project Overview

This repository contains implementations of several computational techniques and algorithms for image processing, feature matching, and deep learning. The work is divided into distinct notebooks, each focusing on a specific task.

---

## Notebooks and Functionalities

### **1. Lab1.ipynb**
#### Gaussian Blur Filter Implementation
- **Purpose**: Apply Gaussian blur to images.
- **Key Functions**:
  - `kernel_gauss(radius, sigma)`: Generates a Gaussian kernel with given radius and sigma.
  - `multiplic_matr(img, kernel, i)`: Multiplies the kernel with the image matrix.
  - `gauss_filter(img, radius, sigma)`: Splits an image into RGB channels, applies Gaussian filtering, and reconstructs the image.
  - `plot_cv_img(input_image, output_image, r, sigma)`: Displays the original and processed images side-by-side.
- **Usage**: Demonstrates various radii and sigma values for Gaussian filtering on sample images.

---

### **2. Lab2.ipynb**
#### Feature Matching with AKAZE Descriptor
- **Purpose**: Match features between two images using the AKAZE descriptor and a custom Hamming distance function.
- **Key Functions**:
  - `match(des, des1)`: Custom feature matching function.
  - `akazu(path, path1)`: Detects features in two images, computes descriptors, matches them, and visualizes the results.
- **Usage**: Compares pairs of images to identify and display matched features.

---

### **3. Lab3_inception_v3.ipynb**
#### Deep Learning with Inception-v3
- **Purpose**: Train a neural network for image classification using the FashionMNIST dataset.
- **Key Features**:
  - Uses a pre-trained Inception-v3 model from `timm` library, modified for the dataset.
  - Includes training and testing loops with performance visualization.
  - Applies t-SNE for dimensionality reduction and visualization of test set features.
- **Key Functions**:
  - `train()`: Performs a single epoch of training.
  - `test()`: Evaluates model performance on the test set.
  - `predict(dataloader)`: Generates predictions and feature embeddings.
- **Visualization**: t-SNE scatter plot for class separability.

---

### **4. Lab3_siamese.ipynb**
#### Siamese Network for Similarity Learning
- **Purpose**: Implement a Siamese network for learning image similarity.
- **Key Features**:
  - Uses the FashionMNIST dataset.
  - Builds a custom architecture for contrastive learning.
  - Applies contrastive loss for training.
- **Usage**: Demonstrates how to train a Siamese network and evaluate similarity between images.

---

## Results and Visualizations
- Gaussian blur: Visualizes original vs. blurred images.
- Feature matching: Displays matched features between two images.
- Deep learning classification: Plots training/test loss and feature clustering.
- Siamese network: Learns and evaluates image similarity.

