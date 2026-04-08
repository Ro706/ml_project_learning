# Feedforward Neural Network (FNN) - Breast Cancer Classification

This project implements a Feedforward Neural Network (also known as a Multi-Layer Perceptron) using PyTorch for binary classification.

## Project Overview

The goal is to predict whether a tumor is malignant or benign based on various clinical features from the Breast Cancer Wisconsin (Diagnostic) dataset.

## Key Components

- **Data Preprocessing**: 
    - Feature scaling using `StandardScaler`.
    - Data splitting (80% train, 20% test).
    - Conversion of NumPy arrays to PyTorch Tensors.
- **Model Architecture**:
    - Input layer: Matching the number of features (30).
    - Hidden Layer 1: 64 neurons with ReLU activation.
    - Hidden Layer 2: 32 neurons with ReLU activation.
    - Output Layer: 1 neuron with Sigmoid activation (for probability).
- **Visualization**:
    - Training Loss Curve.
    - Test Loss Curve.
    - Accuracy growth over epochs.

## How to Run

1.  Navigate to this directory:
    ```bash
    cd deepLearning/fnn
    ```
2.  Install dependencies (shared with the root or use cnn's requirements):
    ```bash
    uv pip install torch scikit-learn matplotlib
    ```
3.  Run the script:
    ```bash
    uv run main.py
    ```

## Results

The script outputs three graphs showing the model's convergence and final accuracy on the test set.
