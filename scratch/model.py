import numpy as np


# Linear regression prediction
def predict(X, w, b):
    """
    X: shape (m, n)
    w: shape (n,)
    b: scalar
    """
    return X @ w + b


# Calculate cost function
def compute_cost(X, y, w, b):
    """
    Mean Squared Error cost function
    """
    m = len(y)
    predictions = predict(X, w, b)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def compute_gradients(X, y, w, b):
    """
    Compute gradients for linear regression
    """
    m = len(y)
    predictions = predict(X, w, b)
    error = predictions - y  # shape (m,)

    dw = (1 / m) * (X.T @ error)  # shape (n_features,)
    db = (1 / m) * np.sum(error)  # scalar

    return dw, db
