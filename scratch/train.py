import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import predict, compute_cost, compute_gradients

# Load dataset
data = pd.read_csv("data/expenses.csv")

# Prepare features and target(label)
X = data[["income", "rent", "utilities", "subscriptions"]].values
y = data["expenses"].values

# Initialize parameters
n_features = X.shape[1]  # number of features
w = np.zeros(n_features)  # weights
b = 0.0  # bias


# Test prediction
y_pred = predict(X, w, b)
print(
    f"First 5 predictions with initial parameters (test/inital dummy run): {y_pred[:5]}"
)
# print(f"X Shape: {X.shape}")
# print(f"w Shape: {w.shape}")

# Test cost function computation
initial_cost = compute_cost(X, y, w, b)
print("Initial cost:", initial_cost)

# Test gradients for linear regression
dw, db = compute_gradients(X, y, w, b)
print("Gradient dw:", dw)
print("Gradient db:", db)


# Hyperparameters
learning_rate = 0.01  # changed from 0.00000001 to 0.01 now that standardization scaling has been used
epochs = 5000

# Standardize features (Standardization scaling method)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

np.save("X_mean.npy", X_mean)
np.save("X_std.npy", X_std)


cost_history = []

# Training loop
for epoch in range(epochs):
    dw, db = compute_gradients(X_scaled, y, w, b)
    w -= learning_rate * dw
    b -= learning_rate * db

    cost = compute_cost(X_scaled, y, w, b)
    cost_history.append(cost)

    # Print cost every 500 epochs
    if epoch % 500 == 0:
        cost = compute_cost(X_scaled, y, w, b)
        print(f"Epoch {epoch}, Cost: {cost:.2f}")

np.save("trained_weights.npy", w)
np.save("trained_bias.npy", b)

print("Learned weights:", w)
print("Learned bias:", b)

# After training, visualize
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), cost_history, color="purple")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost vs Epochs")
plt.show()
