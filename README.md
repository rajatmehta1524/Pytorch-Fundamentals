# PyTorch Deep Learning Repository

## Overview
This repository contains four PyTorch-based deep learning projects, each demonstrating fundamental concepts in deep learning, model training, and evaluation. The projects cover different architectures and datasets, providing hands-on implementation of neural networks for image classification and other tasks.

## Projects Included

### 1. **Basic PyTorch Tensor Operations**
- Introduction to PyTorch tensors.
- Demonstrates basic tensor manipulations, operations, and broadcasting.
- Explores GPU acceleration using CUDA.

### 2. **Feedforward Neural Network on Fashion-MNIST**
- Implements a simple feedforward neural network.
- Trains the model on the Fashion-MNIST dataset.
- Utilizes `nn.Module` to define layers and `cross_entropy` for loss calculation.
- Tracks and visualizes training loss.

### 3. **ResNet-9 on CIFAR-10**
- Implements a convolutional neural network (CNN) based on ResNet-9.
- Applies data augmentation and normalization using `torchvision.transforms`.
- Uses OneCycleLR scheduling for optimal learning rate tuning.
- Visualizes training loss, accuracy, and learning rate trends.
- Includes model evaluation and prediction functions.

### 4. **Linear Classifier for MNIST**
- Defines a simple linear classifier for digit recognition.
- Uses the MNIST dataset with `torchvision.datasets`.
- Implements a structured training loop with validation and evaluation.
- Visualizes accuracy and loss over epochs.
- Saves model parameters using `torch.save()`.

## Installation
To run the projects, ensure you have Python and PyTorch installed. You can set up the environment using the following commands:

```bash
pip install torch torchvision matplotlib
```

## Running the Code
Each project is structured as a standalone script or Jupyter Notebook. To execute, run:

```bash
python script_name.py
```

or, if using Jupyter Notebook:

```bash
jupyter notebook
```

## Contributions
Feel free to contribute by improving model architectures, adding new datasets, or optimizing training pipelines. Fork the repository and submit a pull request!

## License
This repository is open-source and available under the MIT License.

