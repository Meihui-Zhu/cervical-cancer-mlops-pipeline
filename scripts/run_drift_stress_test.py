from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)
from src.data import split_features_target
from src.drift import run_drift_checks
from src.model import NUMERIC_FEATURES


STRESS_TEST_DATE = "2026-05-22"

FEATURES_TO_SHIFT = [
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
]


def create_missingness_stress_batch(
    source_batch: pd.DataFrame,
    features_to_shift: list[str],
    stress_test_date: str = STRESS_TEST_DATE,
    missing_fraction: float = 0.80,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Create a synthetic stress-test batch by increasing missingness in selected features.

    This is used only to demonstrate that the drift monitoring logic can detect
    a data-quality shift. It does not represent a real observed batch.
    """
    rng = np.random.default_rng(random_state)

    stress_batch = source_batch.copy()
    stress_batch["arrival_date"] = stress_test_date

    n_rows = len(stress_batch)
    n_missing = int(np.ceil(n_rows * missing_fraction))

    for feature in features_to_shift:
        if feature not in stress_batch.columns:
            raise ValueError(f"Feature not found in batch: {feature}")

        rows_to_mask = rng.choice(
            stress_batch.index,
            size=n_missing,
            replace=False,
        )

        stress_batch.loc[rows_to_mask, feature] = np.nan

    return stress_batch


def run_missingness_stress_test(
    source_date: str = "2026-05-08",
    stress_test_date: str = STRESS_TEST_DATE,
) -> pd.DataFrame:
    """
    Run a synthetic missingness drift stress test.
    """
    historical_path = PROCESSED_DATA_DIR / "historical_data.csv"
    future_path = PROCESSED_DATA_DIR / "future_data_with_batches.csv"

    if not historical_path.exists() or not future_path.exists():
        raise FileNotFoundError(
            "Processed data files not found. Please run scripts/run_training.py first."
        )

    historical_df = pd.read_csv(historical_path)
    future_df = pd.read_csv(future_path)

    source_batch = future_df[future_df["arrival_date"] == source_date].copy()

    if source_batch.empty:
        available_dates = sorted(future_df["arrival_date"].unique())
        raise ValueError(
            f"No records found for source date {source_date}. "
            f"Available dates are: {available_dates}"
        )

    stress_batch = create_missingness_stress_batch(
        source_batch=source_batch,
        features_to_shift=FEATURES_TO_SHIFT,
        stress_test_date=stress_test_date,
        missing_fraction=0.80,
        random_state=RANDOM_STATE,
    )

    X_historical, _ = split_features_target(historical_df)
    X_stress_batch, _ = split_features_target(stress_batch)

    features = list(X_historical.columns)

    drift_report = run_drift_checks(
        baseline_df=X_historical,
        batch_df=X_stress_batch,
        features=features,
        numeric_features=NUMERIC_FEATURES,
        missingness_threshold=0.20,
        std_multiplier=2.0,
    )

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / f"drift_stress_test_missingness_{stress_test_date}.csv"
    drift_report.to_csv(output_path, index=False)

    flagged = drift_report[drift_report["drift_flag"] == True].copy()

    print("=" * 80)
    print("Synthetic missingness drift stress test completed")
    print("=" * 80)
    print(f"Source batch date: {source_date}")
    print(f"Stress-test batch date: {stress_test_date}")
    print(f"Features with artificially increased missingness: {FEATURES_TO_SHIFT}")
    print(f"Drift report saved to: {output_path}")

    print("\nDrift summary:")
    summary = (
        drift_report
        .groupby("drift_type")["drift_flag"]
        .agg(n_checks="count", n_flagged="sum")
        .reset_index()
    )
    print(summary)

    if flagged.empty:
        print("\nNo drift flags detected. Consider increasing missing_fraction or lowering threshold.")
    else:
        print("\nFlagged drift checks:")
        print(flagged[["feature", "drift_type", "absolute_difference", "threshold"]])

    return drift_report


def main():
    run_missingness_stress_test()


if __name__ == "__main__":
    main()