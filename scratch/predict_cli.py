import numpy as np
from model import predict

# Load trained parameters
w = np.load("trained_weights.npy")
b = np.load("trained_bias.npy")  # b will be a scalar

# Feature names
features = ["income", "rent", "utilities", "subscriptions"]

# Gather user input
user_input = []
for feature in features:
    while True:
        try:
            value = float(input(f"Enter your monthly {feature}: "))
            user_input.append(value)
            break
        except ValueError:
            print("Please enter a valid number.")


# Convert to NumPy array
X_input = np.array(user_input)

# Load scaling parameters
X_mean = np.load("X_mean.npy")
X_std = np.load("X_std.npy")

# Scale input
X_input_scaled = (X_input - X_mean) / X_std

# predict expense
predicted_expense = predict(X_input_scaled, w, b)

print("\nPredicted monthly expenses: ${:.2f}".format(predicted_expense))
