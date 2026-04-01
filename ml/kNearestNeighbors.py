# ==========================================
# Step 1: Import Libraries
# ==========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
# Step 4: Create KNN Model
# ==========================================
model = KNeighborsClassifier(
    n_neighbors=5,   # K value
    metric='euclidean'
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
# Step 8: Accuracy vs K Graph
# ==========================================
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)

plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy (KNN)")
plt.show()
