# AI Course: Machine Learning Implementations

This project contains various machine learning algorithms implemented using Python and `scikit-learn`. It covers fundamental concepts in both **Regression** and **Classification**.

## Project Overview

The repository is structured to provide clear, hands-on examples of common ML models. Each script is self-contained, handling data loading (mostly from `sklearn.datasets`), preprocessing, model training, evaluation, and visualization.

## Getting Started

This project uses `uv` for fast and reliable dependency management.

### Installation

1.  **Install `uv`** (if not already installed):
    ```bash
    pip install uv
    ```

2.  **Create a virtual environment and install dependencies**:
    ```bash
    uv venv
    uv pip install -r ml/requirements.txt
    ```

3.  **Run a script**:
    ```bash
    uv run ml/simple_linear_regression.py
    ```

## Regression vs. Classification

### 1. Regression
Regression is used when the target variable you are trying to predict is **continuous** (a real number).
-   **Goal:** To find the relationship between independent variables and a dependent numerical variable.
-   **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared (R²).
-   **Examples in this project:**
    -   `simple_linear_regression.py`: Predicts California housing prices using a linear relationship.
    -   `polynomial_Regression.py`: Uses higher-degree features to capture non-linear trends in housing data.

### 2. Classification
Classification is used when the target variable consists of **discrete categories** or labels.
-   **Goal:** To assign input data into specific classes.
-   **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
-   **Examples in this project:**
    -   `binaryClassification.py`: Uses Logistic Regression to classify breast cancer as Malignant or Benign.
    -   `DecisionTreeClassification.py`: Classifies Iris flower species using a tree-like model.
    -   `kNearestNeighbors.py`: Classifies data based on the majority label of its nearest neighbors.
    -   `RandomForest.py`: An ensemble method using multiple decision trees for robust classification.
    -   `NaiveBayes.py`: (Currently implements a Random Forest classifier on the breast cancer dataset).

## Project Structure (ml/ directory)

| File | Algorithm | Dataset | Description |
| :--- | :--- | :--- | :--- |
| `simple_linear_regression.py` | Linear Regression | California Housing | Basic linear model for price prediction. |
| `polynomial_Regression.py` | Polynomial Regression | California Housing | Non-linear regression with visualization. |
| `binaryClassification.py` | Logistic Regression | Breast Cancer | Binary classification with full metrics report. |
| `DecisionTreeClassification.py`| Decision Tree | Iris | Visualizes the tree structure and boundaries. |
| `kNearestNeighbors.py` | k-NN | Iris | Includes a K-value optimization graph. |
| `RandomForest.py` | Random Forest | Iris | Ensemble learning with feature importance. |
| `NaiveBayes.py` | Random Forest | Breast Cancer | Random Forest implementation on cancer data. |

## Key Libraries Used

-   **scikit-learn**: For model implementation and datasets.
-   **pandas & numpy**: For data manipulation and numerical operations.
-   **matplotlib & seaborn**: For data visualization and plotting results.

---
