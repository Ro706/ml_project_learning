# Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Fetch dataset
    housing = datasets.fetch_california_housing()
    X = housing.data
    y = housing.target

    # For visualization, pick one feature (median income = column 0)
    X_feature = X[:, [0]]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=42)

    # ---------------- Linear Regression ----------------
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)

    # Metrics
    r2_lin = r2_score(y_test, y_pred_lin)
    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    mse_lin = mean_squared_error(y_test, y_pred_lin)
    rmse_lin = np.sqrt(mse_lin)

    print("=== Linear Regression ===")
    print(f"R2 Score: {r2_lin * 100:.2f}%")
    print(f"MAE: {mae_lin:.2f}")
    print(f"MSE: {mse_lin:.2f}")
    print(f"RMSE: {rmse_lin:.2f}\n")

    # ---------------- Polynomial Regression ----------------
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    # Metrics
    r2_poly = r2_score(y_test, y_pred_poly)
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)

    print("=== Polynomial Regression (Degree 2) ===")
    print(f"R2 Score: {r2_poly * 100:.2f}%")
    print(f"MAE: {mae_poly:.2f}")
    print(f"MSE: {mse_poly:.2f}")
    print(f"RMSE: {rmse_poly:.2f}\n")

    # ---------------- Plots ----------------
    # Actual vs Predicted (Linear vs Polynomial)
    plt.scatter(X_test, y_test, color="blue", alpha=0.5, label="Actual")
    plt.scatter(X_test, y_pred_lin, color="red", alpha=0.5, label="Linear Predicted")
    plt.scatter(X_test, y_pred_poly, color="green", alpha=0.5, label="Polynomial Predicted")
    plt.xlabel("Median Income")
    plt.ylabel("House Price")
    plt.title("Linear vs Polynomial Regression")
    plt.legend()
    plt.show()

    # Residual Plot (Polynomial)
    residuals_poly = y_test - y_pred_poly
    plt.scatter(y_pred_poly, residuals_poly, alpha=0.5, color="purple")
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values (Polynomial)")
    plt.ylabel("Residuals")
    plt.title("Residual Plot (Polynomial Regression)")
    plt.show()

if __name__ == "__main__":
    main()
