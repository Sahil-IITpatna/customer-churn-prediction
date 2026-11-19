"""
Data cleaning and preprocessing pipeline for Customer Churn Prediction.
Strictly prevents data leakage by fitting transformers exclusively on training data.
"""
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    PREPROCESSOR_FILE,
    ID_COL,
    TARGET_COL,
    NUMERICAL_COLS,
    BINARY_CAT_COLS,
    MULTICLASS_CAT_COLS,
    RANDOM_STATE,
    TEST_SIZE
)
from src.data_loader import load_raw_data


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    if ID_COL in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=[ID_COL])

    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
    nan_mask = df_clean["TotalCharges"].isna()
    if nan_mask.sum() > 0:
        df_clean.loc[nan_mask & (df_clean["tenure"] == 0), "TotalCharges"] = 0.0
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["MonthlyCharges"] * df_clean["tenure"])

    df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(int)
    return df_clean


def build_preprocessor_pipeline() -> ColumnTransformer:
    categorical_features = BINARY_CAT_COLS + MULTICLASS_CAT_COLS
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    feature_names = []
    feature_names.extend(NUMERICAL_COLS)
    cat_transformer = preprocessor.named_transformers_["cat"]
    if hasattr(cat_transformer, "get_feature_names_out"):
        categorical_features = BINARY_CAT_COLS + MULTICLASS_CAT_COLS
        cat_names = list(cat_transformer.get_feature_names_out(categorical_features))
        feature_names.extend(cat_names)
    return feature_names


def prepare_datasets(
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)

    X = df_clean.drop(columns=[ID_COL, TARGET_COL])
    y_raw = df_clean[TARGET_COL]
    y = (y_raw == "Yes").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if save_artifacts:
        df_clean.to_csv(CLEANED_DATA_FILE, index=False)
        train_df = X_train.copy()
        train_df[TARGET_COL] = y_train
        train_df.to_csv(TRAIN_DATA_FILE, index=False)

        test_df = X_test.copy()
        test_df[TARGET_COL] = y_test
        test_df.to_csv(TEST_DATA_FILE, index=False)

        joblib.dump(preprocessor, PREPROCESSOR_FILE)

    return X_train, X_test, X_train_transformed, X_test_transformed, y_train, y_test, preprocessor
