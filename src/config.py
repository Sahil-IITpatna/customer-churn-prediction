"""
Configuration settings and paths for Customer Churn Prediction.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset URLs (Official Telco Customer Churn dataset sources)
KAGGLE_DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
]
RAW_DATA_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_telco_churn.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model Artifacts
BEST_MODEL_FILE = MODELS_DIR / "best_churn_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Column Specifications
ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_CAT_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

MULTICLASS_CAT_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

ALL_FEATURE_COLS = NUMERICAL_COLS + BINARY_CAT_COLS + MULTICLASS_CAT_COLS

# Modeling Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DEFAULT_PROB_THRESHOLD = 0.50
OPTIMAL_RECALL_THRESHOLD = 0.35
