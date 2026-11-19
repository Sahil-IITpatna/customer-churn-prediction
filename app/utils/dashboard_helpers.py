"""
Dashboard styling and helper utilities for Streamlit application.
"""
import streamlit as st

def apply_custom_css():
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
