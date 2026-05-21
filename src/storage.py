# src/storage.py

import sqlite3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import (DATABASE_PATH)


def get_connection():
    """Create a SQLite database connection."""
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    """Create database tables if they do not already exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                batch_date TEXT PRIMARY KEY,
                n_records INTEGER,
                n_positive_labels INTEGER,
                positive_rate REAL,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_date TEXT,
                record_id INTEGER,
                biopsy_positive_probability REAL,
                predicted_label INTEGER,
                true_biopsy_label INTEGER,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                UNIQUE(batch_date, record_id, model_version)
            )
        """)


def log_ingestion_batch(batch_date: str, daily_batch: pd.DataFrame):
    """Log metadata for one ingested daily batch."""
    n_records = len(daily_batch)
    n_positive = int(daily_batch["Biopsy"].sum()) if "Biopsy" in daily_batch.columns else None
    positive_rate = float(n_positive / n_records) if n_positive is not None and n_records > 0 else None

    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO ingestion_log (
                batch_date,
                n_records,
                n_positive_labels,
                positive_rate,
                status
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            batch_date,
            n_records,
            n_positive,
            positive_rate,
            "completed",
        ))


def save_predictions_to_database(
    predictions_df: pd.DataFrame,
    model_version: str = "sparse_logistic_v1",
):
    """Save prediction records to the SQLite database."""
    predictions_to_save = predictions_df.copy()
    predictions_to_save["model_version"] = model_version

    # Rename arrival_date to batch_date for the database schema
    predictions_to_save = predictions_to_save.rename(columns={
        "arrival_date": "batch_date"
    })

    rows = predictions_to_save[
        [
            "batch_date",
            "record_id",
            "biopsy_positive_probability",
            "predicted_label",
            "true_biopsy_label",
            "model_version",
        ]
    ].itertuples(index=False, name=None)

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO predictions (
                batch_date,
                record_id,
                biopsy_positive_probability,
                predicted_label,
                true_biopsy_label,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, list(rows))