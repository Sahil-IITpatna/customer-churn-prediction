from setuptools import setup, find_packages

setup(
    name="customer_churn_prediction",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "plotly>=5.15.0",
        "streamlit>=1.30.0",
        "joblib>=1.3.0"
    ],
)
