# ==========================================
# Step 1: Import Libraries
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


# ==========================================
# Step 2: Load Dataset
# ==========================================
iris = load_iris()
X = iris.data # sepal length , sepal width , petal length , petal width
y = iris.target # setosa , versicolor , virginica

# ==========================================
# Step 3: Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# Step 4: Create Model
# ==========================================
model = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=3,
    random_state=42
)


# ==========================================
# Step 5: Train Model
# ==========================================
model.fit(X_train, y_train)


# ==========================================
# Step 6: Predictions & Accuracy
# ==========================================
y_pred = model.predict(X_test)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred))


# ==========================================
# Step 7: Plot Decision Tree
# ==========================================
plt.figure(figsize=(12,8))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree Structure")
plt.show()


# ==========================================
# Step 8: Decision Boundary (2D)
# ==========================================
X_vis = iris.data[:, :2]   # first 2 features
y_vis = iris.target

model_vis = DecisionTreeClassifier(max_depth=3)
model_vis.fit(X_vis, y_vis)

# Mesh grid
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary (Decision Tree)")
plt.show()


# ==========================================
# Step 9: 3D Visualization
# ==========================================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use 3 features
x = X[:, 2]  # petal length
y_axis = X[:, 3]  # petal width
z = X[:, 0]  # sepal length

scatter = ax.scatter(x, y_axis, z, c=y)

ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")
ax.set_zlabel("Sepal Length")

plt.title("3D Scatter Plot - Iris Dataset")
plt.show()
