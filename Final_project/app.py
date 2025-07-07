# xgboost
import streamlit as st
import numpy as np
import pickle
from xgboost import XGBRegressor

# Page config
st.set_page_config(page_title="CLV Predictor", page_icon="🛒", layout="centered")

st.title("🛒 Customer Lifetime Value (CLV) Prediction")
st.write("Enter customer transaction details below to estimate the **Customer Lifetime Value**.")

# Load trained model and scaler
# ✔️ Load XGBoost model saved using model.save_model("clv_model.json")
model = XGBRegressor()
model.load_model(r"C:\College\internship\Sem-6\Final_project\clv_model.json")  # Use .json or .txt as saved

# ✔️ Load scaler saved using pickle
with open(r"C:\College\internship\Sem-6\Final_project\scaler (3).pkl", "rb") as f:
    scaler = pickle.load(f)

# Input fields for the 4 features used in training
col1, col2 = st.columns(2)

with col1:
    recency = st.number_input("Recency (Days Since Last Purchase)", min_value=0, max_value=365, value=30)
    frequency = st.number_input("Purchase Frequency", min_value=1, max_value=100, value=5)

with col2:
    monetary = st.number_input("Total Monetary Value (₹)", min_value=1.0, value=100.0)
    avg_order_value = st.number_input("Average Order Value (₹)", min_value=1.0, value=10.0)

# Predict button
if st.button("Predict CLV"):
    input_data = np.array([[recency, frequency, monetary, avg_order_value]])

    # Scale input data using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Predict CLV using trained XGBoost model
    predicted_log_clv = model.predict(input_scaled)[0]

    # Inverse log1p transform to get actual CLV
    predicted_clv = np.expm1(predicted_log_clv)

    st.success(f"✅ Estimated Customer Lifetime Value: ₹{predicted_clv:.2f}")
