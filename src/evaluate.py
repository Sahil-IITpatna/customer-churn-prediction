"""
Model evaluation module for Customer Churn Prediction.
"""
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

from src.config import (
    MODELS_DIR, REPORTS_DIR, TEST_DATA_FILE, PREPROCESSOR_FILE, TARGET_COL, ID_COL
)


def evaluate_single_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)
        y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())

    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)

    return {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "probabilities": y_probs.tolist(), "predictions": y_pred.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision_vals.tolist(), "recall": recall_vals.tolist()}
    }


def compare_all_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    records = []
    for name, model in models.items():
        res = evaluate_single_model(model, X_test, y_test, threshold=threshold)
        records.append({
            "Model": name,
            "Accuracy": round(res["accuracy"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "F1-Score": round(res["f1"], 4),
            "ROC-AUC": round(res["roc_auc"], 4),
            "PR-AUC": round(res["pr_auc"], 4),
            "True Negatives": res["confusion_matrix"]["tn"],
            "False Positives": res["confusion_matrix"]["fp"],
            "False Negatives": res["confusion_matrix"]["fn"],
            "True Positives": res["confusion_matrix"]["tp"]
        })

    return pd.DataFrame(records).sort_values(by="ROC-AUC", ascending=False)


def analyze_threshold_tradeoffs(model: Any, X_test: np.ndarray, y_test: np.ndarray, cost_false_negative: float = 500.0, cost_false_positive: float = 50.0) -> pd.DataFrame:
    y_probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 17)
    tradeoff_rows = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)

        tradeoff_rows.append({
            "Threshold": round(t, 2), "Precision": round(prec, 4), "Recall": round(rec, 4),
            "F1-Score": round(f1, 4), "False Negatives (Missed Churners)": fn,
            "False Positives (Wasted Offers)": fp, "Total Business Cost ($)": total_cost
        })

    return pd.DataFrame(tradeoff_rows)


if __name__ == "__main__":
    benchmark_df = compare_all_models()
    print("\n--- Model Benchmark Results ---")
    print(benchmark_df.to_string(index=False))

