# random_forest
import streamlit as st
import numpy as np
import joblib

# 🎯 Load trained model and scaler
model = joblib.load(r"C:\College\internship\Sem-6\Final_project\rf_clv_model.pkl")
scaler = joblib.load(r"C:\College\internship\Sem-6\Final_project\scaler.pkl")

# 🛍️ Streamlit page config
st.set_page_config(page_title="CLV Predictor", page_icon="🛒", layout="centered")

st.title("🛒 Customer Lifetime Value (CLV) Prediction")
st.markdown("Enter customer transaction details below to estimate the **Customer Lifetime Value**.")

# 🎛️ Input fields
recency = st.number_input("Recency (Days Since Last Purchase)", min_value=0.0, step=1.0)
frequency = st.number_input("Purchase Frequency", min_value=0.0, step=1.0)
monetary = st.number_input("Total Monetary Value (₹)", min_value=0.0, step=1.0)
aov = st.number_input("Average Order Value (₹)", min_value=0.0, step=1.0)

# 📌 Prediction trigger
if st.button("Predict CLV"):
    try:
        # 🧮 Format input and scale
        input_data = np.array([[recency, frequency, monetary, aov]])
        input_scaled = scaler.transform(input_data)

        # 🔍 Predict and inverse transform
        pred_log = model.predict(input_scaled)
        pred_clv = np.expm1(pred_log[0])  # inverse of log1p

        st.success(f"✅ Estimated Customer Lifetime Value: ₹{pred_clv:.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
