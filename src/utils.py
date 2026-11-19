"""
Utility functions, styling, and plotting helpers for Customer Churn Prediction.
"""
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any

from src.config import BEST_MODEL_FILE, PREPROCESSOR_FILE, CLEANED_DATA_FILE


def apply_custom_css():
    """
    Applies custom modern CSS styling to Streamlit dashboard.
    """
    st.markdown("""
    <style>
        .metric-card {
            background-color: #1E232A;
            border: 1px solid #2E3640;
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 2.1rem;
            font-weight: 700;
            color: #00D26A;
            margin-bottom: 4px;
        }
        .metric-value-danger {
            font-size: 2.1rem;
            font-weight: 700;
            color: #F8312F;
            margin-bottom: 4px;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #9AA0A6;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .risk-badge-high {
            background-color: rgba(231, 76, 60, 0.2);
            color: #E74C3C;
            border: 1px solid #E74C3C;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .risk-badge-med {
            background-color: rgba(243, 156, 18, 0.2);
            color: #F39C12;
            border: 1px solid #F39C12;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .risk-badge-low {
            background-color: rgba(46, 204, 113, 0.2);
            color: #2ECC71;
            border: 1px solid #2ECC71;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, is_danger: bool = False, subtitle: str = ""):
    val_class = "metric-value-danger" if is_danger else "metric-value"
    sub_html = f"<div style='font-size: 0.8rem; color: #80868B;'>{subtitle}</div>" if subtitle else ""
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="{val_class}">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def load_production_pipeline():
    model = joblib.load(BEST_MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    return model, preprocessor


def create_churn_donut_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn Status", "Count"]

    fig = go.Figure(data=[go.Pie(
        labels=counts["Churn Status"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=["#2ECC71", "#E74C3C"]),
        textinfo="label+percent"
    )])
    fig.update_layout(title="Customer Churn Ratio", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_contract_churn_chart(df: pd.DataFrame) -> go.Figure:
    contract_churn = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="Churn Rate (%)")
    fig = px.bar(
        contract_churn, x="Contract", y="Churn Rate (%)", color="Contract",
        text=contract_churn["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        title="Churn Rate by Contract Type", template="plotly_dark",
        color_discrete_sequence=["#E74C3C", "#F39C12", "#2ECC71"]
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis=dict(range=[0, 60]), showlegend=False)
    return fig


def create_payment_churn_chart(df: pd.DataFrame) -> go.Figure:
    pay_churn = df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="Churn Rate (%)").sort_values(by="Churn Rate (%)", ascending=False)
    fig = px.bar(
        pay_churn, x="Churn Rate (%)", y="PaymentMethod", orientation="h", color="PaymentMethod",
        text=pay_churn["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        title="Churn Rate by Payment Method", template="plotly_dark"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis=dict(range=[0, 60]), showlegend=False)
    return fig


def create_tenure_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="tenure", color="Churn", barmode="overlay", nbins=36, opacity=0.75,
        title="Customer Tenure Distribution (Months) by Churn Status",
        color_discrete_map={"No": "#2ECC71", "Yes": "#E74C3C"}, template="plotly_dark"
    )
    return fig


def create_monthly_charges_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="Churn", y="MonthlyCharges", color="Churn", points="outliers",
        title="Monthly Charges ($) Distribution by Churn Status",
        color_discrete_map={"No": "#2ECC71", "Yes": "#E74C3C"}, template="plotly_dark"
    )
    return fig


def create_confusion_matrix_heatmap(cm_dict: Dict[str, int]) -> go.Figure:
    z = [[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]]
    x = ["Predicted Stay (0)", "Predicted Churn (1)"]
    y = ["Actual Stay (0)", "Actual Churn (1)"]
    annotations = [
        dict(x=x[0], y=y[0], text=f"True Negative\n{z[0][0]}", showarrow=False, font=dict(color="white", size=14)),
        dict(x=x[1], y=y[0], text=f"False Positive\n{z[0][1]}", showarrow=False, font=dict(color="white", size=14)),
        dict(x=x[0], y=y[1], text=f"False Negative\n{z[1][0]}", showarrow=False, font=dict(color="white", size=14)),
        dict(x=x[1], y=y[1], text=f"True Positive\n{z[1][1]}", showarrow=False, font=dict(color="white", size=14))
    ]
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion Matrix", annotations=annotations, template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40))
    return fig
