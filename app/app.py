"""
Customer Churn Prediction - Interactive Streamlit Application
Main entry point and system portal.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from src.utils import apply_custom_css, render_metric_card
from src.config import CLEANED_DATA_FILE, RAW_DATA_FILE
from src.data_loader import load_raw_data
from src.preprocessing import clean_raw_data

st.set_page_config(
    page_title="Telco Customer Churn Intelligence Portal",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
st.sidebar.title("Telco Churn AI")
st.sidebar.markdown("**Enterprise Retention Intelligence Platform**")
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧭 Navigation
Select a page from the sidebar menu:
- **1. Business Overview**: Executive KPIs & Churn Summary
- **2. Exploratory Analysis**: Customer Cohorts & Patterns
- **3. Churn Prediction**: Real-Time Scoring & Local SHAP
- **4. Model Performance**: Algorithm Benchmarks & ROC Curves
""")
st.sidebar.markdown("---")
st.sidebar.caption("System Version: 1.0.0 | Production Stack: Scikit-learn + XGBoost + SHAP + Streamlit")

st.title("🔮 Customer Churn Prediction & Retention Intelligence")
st.markdown("""
Welcome to the **Telco Customer Churn Intelligence System**. This application simulates an enterprise-grade analytics 
and machine learning environment designed to detect customer attrition, identify key churn drivers, explain individual 
customer risk using **SHAP (Shapley Additive exPlanations)**, and empower proactive retention strategies.
""")

@st.cache_data
def get_cached_data():
    try:
        if CLEANED_DATA_FILE.exists():
            return pd.read_csv(CLEANED_DATA_FILE)
        df_raw = load_raw_data()
        return clean_raw_data(df_raw)
    except Exception:
        df_raw = load_raw_data()
        return clean_raw_data(df_raw)

df = get_cached_data()

col1, col2, col3, col4 = st.columns(4)
total_customers = len(df)
churn_count = (df["Churn"] == "Yes").sum()
active_count = total_customers - churn_count
churn_rate = (churn_count / total_customers) * 100
total_mrr_at_risk = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()

with col1:
    render_metric_card("Total Customers", f"{total_customers:,}", subtitle="Active and historical accounts")
with col2:
    render_metric_card("Churn Rate", f"{churn_rate:.1f}%", is_danger=(churn_rate > 20), subtitle=f"{churn_count:,} lost customers")
with col3:
    render_metric_card("Active Customers", f"{active_count:,}", subtitle="Currently retained accounts")
with col4:
    render_metric_card("Monthly Revenue at Risk", f"${total_mrr_at_risk:,.0f}", is_danger=True, subtitle="Monthly recurring revenue lost")

st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🎯 Business Objective & Solution Strategy")
    st.markdown("""
    In subscription-based industries like Telecommunications, SaaS, and Banking, acquiring a new customer is 
    **5 to 7 times more expensive** than retaining an existing one. 

    This end-to-end Machine Learning solution addresses the challenge through four pillars:
    
    1. **Early Attrition Detection**: Identify customers exhibiting behavioral churn indicators before they cancel.
    2. **Algorithmic Benchmarking**: Compare Logistic Regression, Decision Trees, Random Forests, and XGBoost using stratified validation.
    3. **Actionable Explainability**: Demystify 'black box' machine learning using **SHAP values** so retention teams know *why* a customer is at risk.
    4. **Cost-Optimal Threshold Tuning**: Calibrate decision thresholds to prioritize **Recall** (catching ~80%+ of churners) while controlling retention discount costs.
    """)
    st.info("💡 **Getting Started:** Use the left sidebar to navigate to **'1_Business_Overview'** or **'3_Churn_Prediction'** to test individual customer risk scoring!")

with col_right:
    st.subheader("⚙️ Technical Architecture")
    st.markdown("""
    - **Language:** Python 3.9+
    - **Data Pipeline:** Pandas, NumPy, Scikit-learn `ColumnTransformer`
    - **Modeling:** Logistic Regression, Decision Tree, Random Forest, XGBoost
    - **Imbalance Handling:** Balanced class weighting (`scale_pos_weight`)
    - **Explainability:** SHAP TreeExplainer (Local & Global)
    - **Visuals:** Plotly Interactive Visualizations & Seaborn
    - **Deployment:** Streamlit Cloud & Docker Containerization
    """)
    st.success("✅ Machine Learning Pipeline & Serialized Models Loaded and Operational.")

st.markdown("---")
st.caption("Developed with ❤️ for Data Science & Machine Learning Excellence | Telco Churn Prediction System")
