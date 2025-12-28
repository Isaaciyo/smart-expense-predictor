import numpy as np
import pandas as pd

np.random.seed(42)

num_samples = 270

income = np.random.randint(2500, 8000, num_samples)
rent = np.random.randint(700, 2500, num_samples)
utilities = np.random.randint(100, 400, num_samples)
subscriptions = np.random.randint(30, 150, num_samples)

# Expenses formula (with noise)
expenses = (
    rent + utilities + subscriptions + np.random.normal(0, 200, num_samples)  # noise
)

data = pd.DataFrame(
    {
        "income": income,
        "rent": rent,
        "utilities": utilities,
        "subscriptions": subscriptions,
        "expenses": expenses,
    }
)

data.to_csv("data/expenses.csv", index=False)
print("Dataset saved as expenses.csv")
