"""
Dashboard Page 1: Business Overview
Executive metrics, churn distribution, and financial exposure.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils import apply_custom_css, render_metric_card, create_churn_donut_chart
from src.config import CLEANED_DATA_FILE
from src.data_loader import load_raw_data
from src.preprocessing import clean_raw_data

st.set_page_config(page_title="Business Overview | Telco Churn", page_icon="📈", layout="wide")
apply_custom_css()

st.title("📈 Page 1: Business Overview & Executive KPIs")
st.markdown("High-level executive metrics monitoring customer retention, attrition patterns, and financial impact.")

@st.cache_data
def get_dataset():
    if CLEANED_DATA_FILE.exists():
        return pd.read_csv(CLEANED_DATA_FILE)
    df_raw = load_raw_data()
    return clean_raw_data(df_raw)

df = get_dataset()

total_customers = len(df)
churn_customers = (df["Churn"] == "Yes").sum()
active_customers = total_customers - churn_customers
churn_rate = (churn_customers / total_customers) * 100
mrr_lost = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()
annual_revenue_risk = mrr_lost * 12

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    render_metric_card("Total Customers", f"{total_customers:,}")
with kpi2:
    render_metric_card("Active Customers", f"{active_customers:,}")
with kpi3:
    render_metric_card("Churned Customers", f"{churn_customers:,}", is_danger=True)
with kpi4:
    render_metric_card("Churn Rate", f"{churn_rate:.2f}%", is_danger=(churn_rate > 20))
with kpi5:
    render_metric_card("Annualized ARR at Risk", f"${annual_revenue_risk:,.0f}", is_danger=True)

st.markdown("---")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("🍩 Overall Customer Churn Distribution")
    donut_fig = create_churn_donut_chart(df)
    st.plotly_chart(donut_fig, use_container_width=True)

with chart_col2:
    st.subheader("💵 Monthly Recurring Revenue (MRR) Breakdown")
    mrr_fig = go.Figure(data=[
        go.Bar(name="Active Revenue", x=["Active"], y=[df[df["Churn"] == "No"]["MonthlyCharges"].sum()], marker_color="#2ECC71"),
        go.Bar(name="Lost Revenue", x=["Churned"], y=[mrr_lost], marker_color="#E74C3C")
    ])
    mrr_fig.update_layout(template="plotly_dark", showlegend=False)
    st.plotly_chart(mrr_fig, use_container_width=True)
