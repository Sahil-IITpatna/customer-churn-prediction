"""
Dashboard Page 3: Real-Time Customer Churn Prediction & Local SHAP Explainability
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

from src.utils import apply_custom_css
from src.config import BEST_MODEL_FILE, PREPROCESSOR_FILE
from src.explainability import ChurnExplainer, CAUSAL_INTERPRETATION_GUIDE
from src.preprocessing import prepare_datasets

st.set_page_config(page_title="Churn Prediction & SHAP | Telco Churn", page_icon="🎯", layout="wide")
apply_custom_css()

st.title("🎯 Page 3: Customer Churn Prediction & Explainability")

@st.cache_resource
def load_ml_assets():
    if not BEST_MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
        from src.train import train_and_persist_all_models
        X_tr, X_te, X_tr_tf, X_te_tf, y_tr, y_te, preproc = prepare_datasets(save_artifacts=True)
        train_and_persist_all_models(X_tr_tf, y_tr, preproc)

    model = joblib.load(BEST_MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    explainer = ChurnExplainer(model=model, preprocessor=preprocessor)
    return model, preprocessor, explainer

model, preprocessor, explainer = load_ml_assets()

with st.form("customer_input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 125.0, 75.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(max(18.8, tenure * monthly_charges)))
    with col3:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

if submitted:
    input_dict = {
        "gender": [gender], "SeniorCitizen": [senior_citizen], "Partner": [partner], "Dependents": [dependents],
        "tenure": [tenure], "PhoneService": [phone_service], "MultipleLines": [multiple_lines],
        "InternetService": [internet_service], "OnlineSecurity": [online_security], "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection], "TechSupport": [tech_support], "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies], "Contract": [contract], "PaperlessBilling": [paperless],
        "PaymentMethod": [payment_method], "MonthlyCharges": [monthly_charges], "TotalCharges": [total_charges]
    }
    cust_df = pd.DataFrame(input_dict)
    X_tf = preprocessor.transform(cust_df)
    prob = float(model.predict_proba(X_tf)[0, 1])

    st.markdown("---")
    st.subheader(f"Predicted Churn Probability: **{prob * 100:.1f}%**")
    if prob >= 0.5:
        st.error("🚨 High Churn Risk!")
    elif prob >= 0.35:
        st.warning("⚠️ Moderate Churn Risk.")
    else:
        st.success("✅ Low Churn Risk (Loyal Customer).")

    expl = explainer.explain_local_customer(cust_df)
    st.markdown("#### Top Factors:")
    for r in expl["top_risk_drivers"][:3]:
        st.write(f"- 🔴 **{r['feature']}**: increases churn probability (+{r['shap_value']:.2f})")
    for p in expl["top_protective_drivers"][:3]:
        st.write(f"- 🟢 **{p['feature']}**: helps retain customer ({p['shap_value']:.2f})")
