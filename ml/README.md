# Machine Learning Implementations (scikit-learn)

This directory contains various classical machine learning algorithms implemented using `scikit-learn`. Each script is a standalone example covering the full pipeline: data loading, preprocessing, training, evaluation, and visualization.

## Setup & Usage

These scripts are managed using `uv`. To run any of them, ensure you have the dependencies installed:

```bash
uv pip install -r requirements.txt
```

Run a specific model:
```bash
uv run ml/simple_linear_regression.py
```

---

## 1. Regression Models

### Simple Linear Regression
- **File**: `simple_linear_regression.py`
- **Dataset**: California Housing
- **Goal**: Predict house prices based on features using a linear relationship.
- **Visuals**: Scatter plot of Actual vs. Predicted values with a regression line.

### Polynomial Regression
- **File**: `polynomial_Regression.py`
- **Dataset**: California Housing (specifically Median Income)
- **Goal**: Capture non-linear relationships by creating polynomial features (Degree 2).
- **Features**: Compares Linear vs. Polynomial performance and includes a **Residual Plot** to check for error patterns.

---

## 2. Classification Models

### Logistic Regression (Binary Classification)
- **File**: `binaryClassification.py`
- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Goal**: Classify tumors as Malignant or Benign.
- **Metrics**: Detailed report including Accuracy, Precision, Recall, F1-Score, and Specificity.
- **Visuals**: Correlation heatmap, confusion matrix heatmap, and top influential features bar chart.

### Decision Tree Classification
- **File**: `DecisionTreeClassification.py`
- **Dataset**: Iris
- **Goal**: Classify flower species using a tree-based logic.
- **Visuals**: 
  - A visual plot of the **Decision Tree structure**.
  - **2D Decision Boundary** map.
  - **3D Scatter Plot** of the dataset features.

### k-Nearest Neighbors (k-NN)
- **File**: `kNearestNeighbors.py`
- **Dataset**: Breast Cancer
- **Goal**: Classify data based on the majority label of the 5 nearest neighbors.
- **Visuals**: Includes a **K-Value vs. Accuracy** optimization graph to help find the best value for `k`.

### Random Forest Classification
- **File**: `RandomForest.py`
- **Dataset**: Iris
- **Goal**: An ensemble of 100 decision trees for more robust classification.
- **Visuals**: **Feature Importance** graph showing which attributes (sepal/petal dimensions) contributed most to the prediction.

### Naive Bayes (Experimental)
- **File**: `NaiveBayes.py`
- **Dataset**: Breast Cancer
- **Current State**: This script currently implements a **Random Forest** approach on the cancer dataset to provide a baseline for binary classification comparison. It includes a confusion matrix and feature importance analysis.

---

## Key Libraries Used
- **scikit-learn**: Core ML algorithms and datasets.
- **pandas & numpy**: Data manipulation and matrix operations.
- **matplotlib & seaborn**: High-quality plotting and statistical visualization.
