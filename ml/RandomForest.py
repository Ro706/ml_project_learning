# ==========================================
# Step 1: Import Libraries
# ==========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# Step 2: Load Dataset
# ==========================================
iris = load_iris()
X = iris.data
y = iris.target


# ==========================================
# Step 3: Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# Step 4: Create Random Forest Model
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=3,           # control overfitting
    random_state=42
)


# ==========================================
# Step 5: Train Model
# ==========================================
model.fit(X_train, y_train)


# ==========================================
# Step 6: Predictions
# ==========================================
y_pred = model.predict(X_test)


# ==========================================
# Step 7: Evaluation
# ==========================================
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================================
# Step 8: Feature Importance Graph
# ==========================================
importances = model.feature_importances_
features = iris.feature_names

plt.figure()
plt.bar(features, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
