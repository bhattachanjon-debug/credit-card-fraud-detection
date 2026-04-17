import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import xgboost as xgb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

@st.cache_resource
def load_model():
    with open("xgboost_fraud_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Credit Card Fraud Detection")
st.write("Enter the transaction details to verify if it's fraudulent.")

col1, col2 = st.columns(2)
with col1:
    v1 = st.number_input("Feature V1", value=0.04)
    v2 = st.number_input("Feature V2", value=0.04)

with col2:
    v3 = st.number_input("Feature V3", value=0.05)
    amount = st.number_input("Transaction Amount", min_value=0.0, value=2.02)

if st.button("Analyze Transaction"):
    try:
        # 1. Create a placeholder for 30 features (Time, V1-V28, Amount)
        input_data = np.zeros((1, 30))
        
        # 2. Map your inputs to the likely indices:
        # Index 1 = V1, Index 2 = V2, Index 3 = V3, Index 29 = Amount
        input_data[0, 1] = v1
        input_data[0, 2] = v2
        input_data[0, 3] = v3
        input_data[0, 29] = amount
        
        # 3. Perform Prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        # 4. Display Results
        st.divider()
        if prediction[0] == 1:
            st.error(f"Potential Fraud Detected!")
            st.metric("Fraud Probability", f"{probability:.2%}")
        else:
            st.success(f"Transaction appears Normal.")
            st.metric("Fraud Probability", f"{probability:.2%}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Tip: Ensure your model was trained on 30 features (Time, V1-V28, Amount).")
