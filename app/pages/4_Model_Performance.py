"""
Dashboard Page 4: Model Performance & Algorithmic Benchmarks
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import joblib

from src.utils import apply_custom_css
from src.config import MODELS_DIR, TEST_DATA_FILE, PREPROCESSOR_FILE, TARGET_COL, ID_COL
from src.evaluate import compare_all_models
from src.preprocessing import prepare_datasets

st.set_page_config(page_title="Model Performance | Telco Churn", page_icon="🏆", layout="wide")
apply_custom_css()

st.title("🏆 Page 4: Machine Learning Model Benchmarks")

@st.cache_resource
def load_eval_data_and_models():
    if not TEST_DATA_FILE.exists() or not PREPROCESSOR_FILE.exists():
        from src.train import train_and_persist_all_models
        X_tr, X_te, X_tr_tf, X_te_tf, y_tr, y_te, preproc = prepare_datasets(save_artifacts=True)
        train_and_persist_all_models(X_tr_tf, y_tr, preproc)

    test_df = pd.read_csv(TEST_DATA_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    X_test_raw = test_df.drop(columns=[TARGET_COL])
    if ID_COL in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[ID_COL])
    y_test = test_df[TARGET_COL].values
    X_test_tf = preprocessor.transform(X_test_raw)

    model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    models = {}
    for name in model_names:
        slug = name.lower().replace(" ", "_")
        path = MODELS_DIR / f"{slug}.pkl"
        if path.exists():
            models[name] = joblib.load(path)

    return test_df, preprocessor, X_test_tf, y_test, models

test_df, preprocessor, X_test_tf, y_test, models = load_eval_data_and_models()

comp_df = compare_all_models(models, X_test_tf, y_test, threshold=0.50)
st.dataframe(comp_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]], use_container_width=True)
