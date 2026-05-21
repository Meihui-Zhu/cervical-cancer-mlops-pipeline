# scripts/run_daily_inference.py

import argparse
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.config import (
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
)
from src.data import split_features_target


def run_daily_inference(batch_date: str):
    """
    Run inference for one simulated daily batch.
    """
    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)

    future_data_path = PROCESSED_DATA_DIR / "future_data_with_batches.csv"

    if not future_data_path.exists():
        raise FileNotFoundError(
            f"{future_data_path} not found. Please run scripts/run_training.py first."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Please run scripts/run_training.py first."
        )

    # Load simulated future data with daily arrival dates
    future_df = pd.read_csv(future_data_path)

    # Select one daily batch
    daily_batch = future_df[future_df["arrival_date"] == batch_date].copy()

    if daily_batch.empty:
        available_dates = sorted(future_df["arrival_date"].unique())
        raise ValueError(
            f"No records found for date {batch_date}. "
            f"Available dates are: {available_dates}"
        )

    # Load fitted model pipeline
    model = joblib.load(MODEL_PATH)

    # Split features and target.
    # The target is not used for prediction, but is kept separately for offline evaluation.
    X_batch, y_batch = split_features_target(daily_batch)

    # Predict probabilities and labels
    risk_scores = model.predict_proba(X_batch)[:, 1]
    predicted_labels = model.predict(X_batch)

    # Build prediction output
    predictions_df = pd.DataFrame({
        "arrival_date": daily_batch["arrival_date"].values,
        "record_id": daily_batch.index,
        "biopsy_positive_probability": risk_scores,
        "predicted_label": predicted_labels,
        "true_biopsy_label": y_batch.values,
    })

    output_path = PREDICTIONS_DIR / f"predictions_{batch_date}.csv"
    predictions_df.to_csv(output_path, index=False)

    print(f"Generated predictions for {batch_date}")
    print(f"Number of records: {len(predictions_df)}")
    print(f"Saved predictions to: {output_path}")

    print("\nPrediction summary:")
    print(predictions_df["biopsy_positive_probability"].describe())

    print("\nPredicted label counts:")
    print(predictions_df["predicted_label"].value_counts())

    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Run inference for one simulated daily batch."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Arrival date to process, for example 2026-05-08 (YYYY-MM-DD).",
    )

    args = parser.parse_args()
    run_daily_inference(args.date)


if __name__ == "__main__":
    main()