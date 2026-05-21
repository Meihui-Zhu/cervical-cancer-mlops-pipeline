import argparse
from pathlib import Path

import pandas as pd
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.data import split_features_target
from src.drift import run_drift_checks
from src.model import NUMERIC_FEATURES


def run_drift_check(batch_date: str) -> pd.DataFrame:
    """
    Run drift checks for one simulated daily batch.
    """
    historical_path = PROCESSED_DATA_DIR / "historical_data.csv"
    future_path = PROCESSED_DATA_DIR / "future_data_with_batches.csv"

    if not historical_path.exists() or not future_path.exists():
        raise FileNotFoundError(
            "Processed data files not found. Please run scripts/run_training.py first."
        )

    historical_df = pd.read_csv(historical_path)
    future_df = pd.read_csv(future_path)

    daily_batch = future_df[future_df["arrival_date"] == batch_date].copy()

    if daily_batch.empty:
        available_dates = sorted(future_df["arrival_date"].unique())
        raise ValueError(
            f"No records found for date {batch_date}. "
            f"Available dates are: {available_dates}"
        )

    X_historical, _ = split_features_target(historical_df)
    X_batch, _ = split_features_target(daily_batch)

    features = list(X_historical.columns)

    drift_report = run_drift_checks(
        baseline_df=X_historical,
        batch_df=X_batch,
        features=features,
        numeric_features=NUMERIC_FEATURES,
        missingness_threshold=0.20,
        std_multiplier=2.0,
    )

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / f"drift_report_{batch_date}.csv"
    drift_report.to_csv(output_path, index=False)

    print(f"Drift report saved to: {output_path}")

    print("\nDrift summary:")
    summary = (
        drift_report
        .groupby("drift_type")["drift_flag"]
        .agg(n_checks="count", n_flagged="sum")
        .reset_index()
    )
    print(summary)

    flagged = drift_report[drift_report["drift_flag"] == True]

    if flagged.empty:
        print("\nNo drift flags detected.")
    else:
        print("\nFlagged drift checks:")
        print(flagged[["feature", "drift_type", "absolute_difference", "threshold"]])

    return drift_report


def run_all_drift_checks() -> None:
    """
    Run drift checks for all available simulated daily batches.
    """
    future_path = PROCESSED_DATA_DIR / "future_data_with_batches.csv"

    if not future_path.exists():
        raise FileNotFoundError(
            "Future data file not found. Please run scripts/run_training.py first."
        )

    future_df = pd.read_csv(future_path)
    batch_dates = sorted(future_df["arrival_date"].unique())

    print(f"Found {len(batch_dates)} batch dates:")
    print(batch_dates)

    all_reports = []

    for batch_date in batch_dates:
        print("\n" + "=" * 80)
        print(f"Running drift check for batch date: {batch_date}")
        print("=" * 80)

        drift_report = run_drift_check(batch_date)
        drift_report["batch_date"] = batch_date
        all_reports.append(drift_report)

    combined_report = pd.concat(all_reports, ignore_index=True)

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    combined_output_path = REPORTS_DIR / "drift_report_all_batches.csv"
    combined_report.to_csv(combined_output_path, index=False)

    print("\n" + "=" * 80)
    print("All drift checks completed.")
    print(f"Combined drift report saved to: {combined_output_path}")
    print("=" * 80)




def main():
    parser = argparse.ArgumentParser(
        description="Run drift checks for one simulated daily batch."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Arrival date to process, for example 2026-05-08.",
    )

    args = parser.parse_args()
    if args.date.lower() == "all":
        run_all_drift_checks()
    else:
        run_drift_check(args.date)


if __name__ == "__main__":
    main()