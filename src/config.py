# src/config.py

from pathlib import Path

RANDOM_STATE = 42

TARGET_COL = "Biopsy"
TARGET_COLS = ["Hinselmann", "Schiller", "Citology", "Biopsy"]

TEST_SIZE = 0.30
N_DAILY_BATCHES = 14
START_DATE = "2026-05-08"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_EXTERNAL_PATH = PROJECT_ROOT / "data" / "external" / "risk_factors_cervical_cancer.csv"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "biopsy_sparse_logistic_pipeline.joblib"