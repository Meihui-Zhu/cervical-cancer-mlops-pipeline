# src/data.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TARGET_COL,
    TARGET_COLS,
    TEST_SIZE,
    N_DAILY_BATCHES,
    START_DATE,
)


def load_dataset(path):
    """Load the original UCI Cervical Cancer Risk Factors dataset."""
    df = pd.read_csv(path)

    # UCI missing values are encoded as "?"
    df = df.replace("?", np.nan)

    # Convert all columns to numeric where possible
    df = df.apply(pd.to_numeric)

    # Add a synthetic record identifier because the original dataset has no patient ID.
    df.insert(0, "record_id", range(1, len(df) + 1))

    return df


def create_historical_future_split(df):
    """
    Split the static dataset into historical records and simulated future records.

    The split is stratified by the Biopsy outcome to preserve the rare positive class.
    """
    historical_df, future_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )

    historical_df = historical_df.reset_index(drop=True).copy()
    future_df = future_df.reset_index(drop=True).copy()

    return historical_df, future_df


def assign_daily_batches(
    future_df,
    start_date=START_DATE,
    n_days=N_DAILY_BATCHES,
    random_state=RANDOM_STATE,
):
    """
    Assign synthetic arrival dates to future records to simulate daily batches.
    """

    # shuffle the future_df
    future_df = future_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True).copy()

    dates = pd.date_range(start=start_date, periods=n_days, freq="D").strftime("%Y-%m-%d")
    batch_indices = np.array_split(np.arange(len(future_df)), n_days)

    future_df["arrival_date"] = ""

    for date, indices in zip(dates, batch_indices):
        future_df.loc[indices, "arrival_date"] = date

    return future_df


def split_features_target(df):
    """
    Split dataframe into feature matrix X and target vector y.

    Other target columns are excluded from features to avoid target leakage.
    """

    X = df.drop(columns=TARGET_COLS, errors="ignore").copy()

    # Drop synthetic record identifier
    X = X.drop(columns=["record_id"], errors="ignore")

    # Remove metadata columns if present
    if "arrival_date" in X.columns:
        X = X.drop(columns=["arrival_date"], errors="ignore")

    y = df[TARGET_COL].copy()

    return X, y