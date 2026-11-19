"""
Model training module for Customer Churn Prediction.
"""
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    MODELS_DIR,
    BEST_MODEL_FILE,
    METADATA_FILE,
    RANDOM_STATE,
    CV_FOLDS
)
from src.preprocessing import prepare_datasets, get_feature_names_out


def get_models(scale_pos_weight: float = 2.76) -> Dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=20, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE
        )
    }
    return models


def cross_validate_models(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = []

    for name, model in models.items():
        cv_res = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        results.append({
            "Model": name,
            "CV_Accuracy_Mean": cv_res["test_accuracy"].mean(),
            "CV_Precision_Mean": cv_res["test_precision"].mean(),
            "CV_Recall_Mean": cv_res["test_recall"].mean(),
            "CV_F1_Mean": cv_res["test_f1"].mean(),
            "CV_ROC_AUC_Mean": cv_res["test_roc_auc"].mean(),
        })

    return pd.DataFrame(results)


def train_and_persist_all_models(X_train: np.ndarray, y_train: np.ndarray, preprocessor: Any) -> Dict[str, Any]:
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = float(neg_count) / max(1.0, float(pos_count))

    models = get_models(scale_pos_weight=scale_pos_weight)
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        slug = name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"{slug}.pkl"
        joblib.dump(model, model_path)

    best_model_name = "XGBoost"
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, BEST_MODEL_FILE)

    feature_names = get_feature_names_out(preprocessor)
    metadata = {
        "best_model_name": best_model_name,
        "scale_pos_weight": scale_pos_weight,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "random_state": RANDOM_STATE
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    return trained_models


if __name__ == "__main__":
    X_tr, X_te, X_tr_tf, X_te_tf, y_tr, y_te, preprocessor = prepare_datasets()
    fitted_models = train_and_persist_all_models(X_tr_tf, y_tr, preprocessor)
    print("All models successfully trained and serialized!")
