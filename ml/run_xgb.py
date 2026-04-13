import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. Load the Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Split the Data
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the XGBoost Model
# We use XGBClassifier for classification tasks
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='binary:logistic',
    random_state=42
)

print("Training XGBoost model...")
model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test)

# 5. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Feature Importance (Bonus Engineering Insight)
# XGBoost can tell us which features mattered most
import matplotlib.pyplot as plt

# Get feature importance
importances = model.feature_importances_
feature_imp = pd.Series(importances, index=data.feature_names).sort_values(ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_imp.head(5))
