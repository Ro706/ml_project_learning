# AI Course: Machine Learning & Deep Learning Implementations

This project contains various machine learning and deep learning algorithms implemented using Python, `scikit-learn`, and `PyTorch`. It covers fundamental concepts ranging from classical **Regression** and **Classification** to **Neural Networks**.

## Project Overview

The repository is structured to provide clear, hands-on examples of common AI models. Each script is self-contained, handling data loading, preprocessing, model training, evaluation, and visualization.

## Current Progress

- [x] **Classical Machine Learning (`ml/`)**
    - [x] Simple Linear Regression
    - [x] Polynomial Regression
    - [x] Logistic Regression (Binary Classification)
    - [x] Decision Tree Classification
    - [x] k-Nearest Neighbors (k-NN)
    - [x] Random Forest Classification
    - [x] Naive Bayes (implemented via Random Forest)
- [x] **Deep Learning (`deepLearning/`)**
    - [x] Convolutional Neural Network (CNN) for MNIST
    - [x] Feedforward Neural Network (FNN) for Binary Classification
- [ ] **Advanced Topics** (Planned)
    - [ ] Recurrent Neural Networks (RNN)
    - [ ] Natural Language Processing (NLP) basics
    - [ ] Reinforcement Learning introduction

## Getting Started

This project uses `uv` for fast and reliable dependency management.

### Installation

1.  **Install `uv`**:
    ```bash
    pip install uv
    ```

2.  **Create a virtual environment**:
    ```bash
    uv venv
    ```

3.  **Install dependencies**:
    *   For Classical ML: `uv pip install -r ml/requirements.txt`
    *   For Deep Learning: `uv pip install -r deepLearning/cnn/requirements.txt`

### Running Scripts

*   **Classical ML**: `uv run ml/simple_linear_regression.py`
*   **Deep Learning (CNN)**: `cd deepLearning/cnn && uv run main.py`
*   **Deep Learning (FNN)**: `cd deepLearning/fnn && uv run main.py`

## Project Structure

### 1. Machine Learning (`ml/`)
| File | Algorithm | Dataset | Description |
| :--- | :--- | :--- | :--- |
| `simple_linear_regression.py` | Linear Regression | California Housing | Basic linear model for price prediction. |
| `polynomial_Regression.py` | Polynomial Regression | California Housing | Non-linear regression with visualization. |
| `binaryClassification.py` | Logistic Regression | Breast Cancer | Binary classification with full metrics report. |
| `DecisionTreeClassification.py`| Decision Tree | Iris | Visualizes the tree structure and boundaries. |
| `kNearestNeighbors.py` | k-NN | Iris | Includes a K-value optimization graph. |
| `RandomForest.py` | Random Forest | Iris | Ensemble learning with feature importance. |
| `NaiveBayes.py` | Random Forest | Breast Cancer | Random Forest implementation on cancer data. |

### 2. Deep Learning (`deepLearning/`)
| Directory | Architecture | Dataset | Description |
| :--- | :--- | :--- | :--- |
| `cnn/` | CNN | MNIST | 2D Convolutional layers for digit recognition. |
| `fnn/` | FNN | Breast Cancer | Multi-layer perceptron for binary classification. |

## Key Libraries Used

-   **scikit-learn**: For classical ML and datasets.
-   **PyTorch**: For building and training neural networks.
-   **pandas & numpy**: For data manipulation and numerical operations.
-   **matplotlib & seaborn**: For data visualization.
-   **opencv-python**: Used in CNN for image processing tasks.

---
