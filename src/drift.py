import numpy as np
import pandas as pd

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import NUMERIC_FEATURES


def calculate_missingness_drift(
    baseline_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Compare missingness rates between historical baseline data and one incoming batch.

    A feature is flagged if the absolute difference in missingness rate exceeds
    the specified threshold.
    """
    rows = []

    for feature in features:
        baseline_missing_rate = baseline_df[feature].isna().mean()
        batch_missing_rate = batch_df[feature].isna().mean()
        absolute_difference = abs(batch_missing_rate - baseline_missing_rate)

        rows.append({
            "feature": feature,
            "drift_type": "missingness",
            "baseline_missing_rate": baseline_missing_rate,
            "batch_missing_rate": batch_missing_rate,
            "absolute_difference": absolute_difference,
            "threshold": threshold,
            "drift_flag": absolute_difference > threshold,
        })

    return pd.DataFrame(rows)


def calculate_numeric_feature_drift(
    baseline_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    numeric_features: list[str] = NUMERIC_FEATURES,
    std_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Compare numeric feature means between historical baseline data and one incoming batch.

    A feature is flagged if the batch mean differs from the baseline mean by more
    than std_multiplier times the baseline standard deviation.
    """
    rows = []

    for feature in numeric_features:
        baseline_values = baseline_df[feature].dropna()
        batch_values = batch_df[feature].dropna()

        baseline_mean = baseline_values.mean()
        batch_mean = batch_values.mean()

        baseline_std = baseline_values.std()

        if pd.isna(baseline_std) or baseline_std == 0 or batch_values.empty:
            threshold_value = np.nan
            absolute_difference = abs(batch_mean - baseline_mean) if not pd.isna(batch_mean) else np.nan
            drift_flag = False
            status = "not_evaluated"
        else:
            threshold_value = std_multiplier * baseline_std
            absolute_difference = abs(batch_mean - baseline_mean)
            drift_flag = absolute_difference > threshold_value
            status = "evaluated"

        rows.append({
            "feature": feature,
            "drift_type": "numeric_mean_shift",
            "baseline_mean": baseline_mean,
            "batch_mean": batch_mean,
            "baseline_std": baseline_std,
            "absolute_difference": absolute_difference,
            "threshold": threshold_value,
            "std_multiplier": std_multiplier,
            "drift_flag": drift_flag,
            "status": status,
        })

    return pd.DataFrame(rows)


def run_drift_checks(
    baseline_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    features: list[str],
    numeric_features: list[str] = NUMERIC_FEATURES,
    missingness_threshold: float = 0.20,
    std_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Run missingness and numeric feature drift checks for one incoming batch.
    """
    missingness_report = calculate_missingness_drift(
        baseline_df=baseline_df,
        batch_df=batch_df,
        features=features,
        threshold=missingness_threshold,
    )

    numeric_report = calculate_numeric_feature_drift(
        baseline_df=baseline_df,
        batch_df=batch_df,
        numeric_features=numeric_features,
        std_multiplier=std_multiplier,
    )

    drift_report = pd.concat(
        [missingness_report, numeric_report],
        ignore_index=True,
        sort=False,
    )

    return drift_report