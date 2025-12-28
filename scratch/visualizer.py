import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import predict

# Load data
data = pd.read_csv("data/expenses.csv")
X = data[["income", "rent", "utilities", "subscriptions"]].values
y = data["expenses"].values

# Standardize features (Standardization scaling method)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# Use learned weights (use your trained w and b)
# w = np.array([0.02231622, 1.0584854, 0.50751705, 0.1159585])
# b = 0.0008529092043912532

# Better way of getting the values from train.py
w = np.load("trained_weights.npy")
b = np.load("trained_bias.npy")

y_pred = predict(X_scaled, w, b)

plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color="blue", alpha=0.6)
plt.plot(
    [y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2
)  # perfect prediction line
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")
plt.title("Predicted vs Actual Expenses")
plt.show()
