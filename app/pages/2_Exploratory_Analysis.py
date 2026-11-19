"""
Dashboard Page 2: Exploratory Analysis (EDA)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils import apply_custom_css
from src.config import CLEANED_DATA_FILE
from src.data_loader import load_raw_data
from src.preprocessing import clean_raw_data
from src.utils import (
    create_contract_churn_chart,
    create_payment_churn_chart,
    create_tenure_distribution_chart,
    create_monthly_charges_chart
)

st.set_page_config(page_title="Exploratory Analysis | Telco Churn", page_icon="🔍", layout="wide")
apply_custom_css()

st.title("🔍 Page 2: Exploratory Data Analysis & Cohort Insights")

@st.cache_data
def get_dataset():
    if CLEANED_DATA_FILE.exists():
        return pd.read_csv(CLEANED_DATA_FILE)
    df_raw = load_raw_data()
    return clean_raw_data(df_raw)

df = get_dataset()

tab1, tab2, tab3 = st.tabs(["📑 Contract Patterns", "⏳ Tenure Cohorts", "💳 Charges & Payments"])

with tab1:
    st.subheader("Contract Duration & Churn")
    fig_contract = create_contract_churn_chart(df)
    st.plotly_chart(fig_contract, use_container_width=True)

with tab2:
    st.subheader("Tenure Distribution")
    fig_tenure = create_tenure_distribution_chart(df)
    st.plotly_chart(fig_tenure, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_monthly_charges_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(create_payment_churn_chart(df), use_container_width=True)
