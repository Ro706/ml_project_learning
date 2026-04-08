# Convolutional Neural Network (CNN) - MNIST

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

## Architecture

The model (`CNN` class in `main.py`) consists of:
- **Feature Extraction**: Two convolutional layers with ReLU activation and Max-Pooling.
    - Conv1: 1 input channel, 32 output channels, 3x3 kernel.
    - Conv2: 32 input channels, 64 output channels, 3x3 kernel.
- **Classification**: A fully connected (dense) network.
    - Flatten layer.
    - Linear layer: 128 neurons with ReLU.
    - Output layer: 10 neurons (one for each digit 0-9).

## Dataset

- **MNIST**: A collection of 70,000 grayscale images of handwritten digits.
- **Preprocessing**: Images are converted to tensors and normalized to a range of [-1, 1].

## Training

The script trains for 5 epochs using:
- **Optimizer**: Adam (learning rate = 0.001).
- **Loss Function**: Cross-Entropy Loss.
- **Output**: The trained model state is saved as `mnist_cnn.pth`.

## How to Run

1.  Navigate to this directory:
    ```bash
    cd deepLearning/cnn
    ```
2.  Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```
3.  Run the training script:
    ```bash
    uv run main.py
    ```
