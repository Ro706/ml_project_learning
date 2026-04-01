# ==============================
# 1. Import Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==============================
# 2. Load Dataset
# ==============================
data = load_breast_cancer() #loading the dataset : breast_cancer()

X = data.data # Feature
y = data.target #Target

# Convert to DataFrame for better understanding
print(data.feature_names)

df = pd.DataFrame(X, columns=data.feature_names) 
df['target'] = y # Target which tell whether is 0 & 1 --> (malignant (cancer) & Benign (No cancer)) 

print("Dataset Shape:", df.shape) # It return the [rows * columns]
print("\nFirst 5 Rows:\n", df.head()) # top 5 rows data

# ==============================
# 3. Data Analysis
# ==============================

# Class distribution
print("\nClass Distribution:\n", df['target'].value_counts()) #This count the unique value in dataframe -> 0 : 212 & 1 : 357 

# Plot class distribution
sns.countplot(x='target', data=df) # Showing the Target class distribution 
plt.title("Class Distribution (0 = Malignant, 1 = Benign)")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6)) 
sns.heatmap(df.corr(), cmap='coolwarm') #correlation matrix as heatmap [max:1 min: (-0.79)]
plt.title("Feature Correlation")
plt.show()

# ==============================
# 4. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) # split the data into two part X -> contain all the features and Y -> which contain the target data 

# ==============================
# 5. Feature Scaling
# ==============================
scaler = StandardScaler() # This is use to normalize features before feeding them into a machine learning model 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# ==============================
# 6. Train Model
# ==============================
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# ==============================
# 7. Predictions
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 8. Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, y_pred) # preparing confusion matrix
print("\nConfusion Matrix:\n", cm)

# Extract values
TN, FP, FN, TP = cm.ravel()

# ==============================
# 9. Evaluation Metrics (Manual)
# ==============================
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
specificity = TN / (TN + FP)

print("\nEvaluation Metrics:")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"Specificity  : {specificity:.4f}")

# ==============================
# 10. Sklearn Report
# ==============================
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 11. Visualize Confusion Matrix
# ==============================
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 12. Feature Importance
# ==============================
coefficients = pd.DataFrame(
    model.coef_[0],
    index=data.feature_names,
    columns=["Coefficient"]
)

print("\nTop Features:\n", coefficients.sort_values(by="Coefficient", ascending=False).head(10))

# Plot top features
coefficients.sort_values(by="Coefficient").tail(10).plot(kind='barh')
plt.title("Top Influential Features")
plt.show()
