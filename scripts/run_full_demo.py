# scripts/run_full_demo.py

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import shutil

from src.config import PROCESSED_DATA_DIR, PREDICTIONS_DIR
from scripts.run_training import main as run_training_main
from scripts.run_daily_inference import run_daily_inference
from scripts.run_drift_check import run_drift_check

from src.config import (
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    DATABASE_PATH
)



def clean_generated_outputs():
    """
    Remove pipeline-generated outputs from previous demo runs.

    This does not remove data/external/, which contains the original dataset.
    """
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
        print(f"Removed existing database: {DATABASE_PATH}")

    output_dirs = [
        PROCESSED_DATA_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]

    for directory in output_dirs:
        directory = Path(directory)
        if directory.exists():
            shutil.rmtree(directory)
            print(f"Removed existing directory: {directory}")

        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created clean directory: {directory}")




def main():
    """
    Run the full end-to-end demo.

    This script:
    1. Cleans generated outputs from previous runs.
    2. Trains and saves the final model using historical data.
    3. Loads the simulated future data with daily arrival dates.
    4. Runs daily inference for every available batch date.
    """

    print("=" * 80)
    print("Step 0: Cleaning generated outputs from previous runs")
    print("=" * 80)

    clean_generated_outputs()

    print("=" * 80)
    print("Step 1: Training final model and generating evaluation reports")
    print("=" * 80)

    run_training_main()

    future_data_path = PROCESSED_DATA_DIR / "future_data_with_batches.csv"

    if not future_data_path.exists():
        raise FileNotFoundError(
            f"{future_data_path} not found after training step."
        )

    future_df = pd.read_csv(future_data_path)

    batch_dates = sorted(future_df["arrival_date"].unique())

    print("\n" + "=" * 80)
    print("Step 2: Running daily inference for simulated incoming batches")
    print("=" * 80)

    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)

    for batch_date in batch_dates:
        print("\n" + "-" * 80)
        print(f"Processing daily batch: {batch_date}")
        print("-" * 80)

        run_daily_inference(batch_date)
        run_drift_check(batch_date)

    print("\n" + "=" * 80)
    print("Full demo completed successfully.")
    print("=" * 80)

    print(f"\nGenerated prediction files are saved in: {PREDICTIONS_DIR}")
    print("Generated model and reports are saved in: models/ and reports/")


if __name__ == "__main__":
    main()