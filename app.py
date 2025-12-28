import streamlit as st
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt  (Not needed since plotly is more dynamic)
import plotly.graph_objects as go
from scratch.model import predict

# Load trained parameters
w = np.load("trained_weights.npy")
b = np.load("trained_bias.npy")

# Load scaling parameters since I scaled features
X_mean = np.load("X_mean.npy")
X_std = np.load("X_std.npy")

# Set page
st.set_page_config(page_title="Smart Expense Predictor Dashboard", layout="wide")
st.title("💰 Smart Expense Predictor Dashboard")
st.write("Predict your monthly expenses and explore feature impacts.")

# Sidebar for user input
st.sidebar.header("Input Your Monthly Financial Info")
income = st.sidebar.number_input("Monthly Income ($)", 0, 20000, 5000, step=100)
rent = st.sidebar.number_input("Monthly Rent ($)", 0, 5000, 1200, step=50)
utilities = st.sidebar.number_input("Monthly Utilities ($)", 0, 1000, 300, step=25)
subscriptions = st.sidebar.number_input(
    "Monthly Subscriptions ($)", 0, 500, 100, step=10
)

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.markdown(
    """
    Built by: **Isidahome-Iyobebe Isaac**  
    [GitHub Repository](https://github.com/Isaaciyo/smart-expense-predictor)
    """
)

X_input = np.array([income, rent, utilities, subscriptions])
X_input = (X_input - X_mean) / X_std

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Predict Expenses"):
    predicted_expense = predict(X_input, w, b)
    st.success(f"Predicted Monthly Expenses: ${predicted_expense:.2f}")

    # Add to session history
    st.session_state.history.append(
        {
            "Income": income,
            "Rent": rent,
            "Utilities": utilities,
            "Subscriptions": subscriptions,
            "Predicted Expenses": predicted_expense,
        }
    )

# Show prediction history
if st.session_state.history:
    st.subheader("📊 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    # Download CSV
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
    )


# Feature weights visualization with tooltips
st.subheader("📈 Feature Impact on Expenses")
feature_names = ["Income", "Rent", "Utilities", "Subscriptions"]
feature_values = w

fig = go.Figure(
    data=[
        go.Bar(
            x=feature_names,
            y=feature_values,
            text=[f"{val:.2f}" for val in feature_values],  # show value on top of bars
            hoverinfo="y",  # only show the value on hover
            marker_color="skyblue",
        )
    ]
)

fig.update_layout(
    yaxis_title="Weight Value", title="Learned Feature Weights", template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Display bias
st.write(f"Bias (baseline expense): {b:.2f}")
