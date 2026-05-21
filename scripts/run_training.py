# scripts/run_training.py

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))


import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

from src.config import (
    DATA_EXTERNAL_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)
from src.data import (
    assign_daily_batches,
    create_historical_future_split,
    load_dataset,
    split_features_target,
)
from src.model import build_sparse_logistic_pipeline


def main():
    # Create output directories
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Load and split data
    df = load_dataset(DATA_EXTERNAL_PATH)

    historical_df, future_df = create_historical_future_split(df)
    future_df = assign_daily_batches(future_df)

    # Save simulated data splits
    historical_df.to_csv(PROCESSED_DATA_DIR / "historical_data.csv", index=False)
    future_df.to_csv(PROCESSED_DATA_DIR / "future_data_with_batches.csv", index=False)

    # Prepare features and target
    X_historical, y_historical = split_features_target(historical_df)

    # Build final pipeline
    pipeline = build_sparse_logistic_pipeline()

    # Cross-validation
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = cross_validate(
        pipeline,
        X_historical,
        y_historical,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    cv_summary = pd.DataFrame({
        metric: [
            cv_results[f"test_{metric}"].mean(),
            cv_results[f"test_{metric}"].std(),
        ]
        for metric in scoring
    }, index=["mean", "std"]).T

    cv_summary.to_csv(REPORTS_DIR / "cv_results_sparse_logistic.csv")

    print("\nCross-validation summary:")
    print(cv_summary)

    # Cross-validated confusion matrix
    y_pred_cv = cross_val_predict(
        pipeline,
        X_historical,
        y_historical,
        cv=cv,
        method="predict",
    )

    cm = confusion_matrix(y_historical, y_pred_cv)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual negative", "Actual positive"],
        columns=["Predicted negative", "Predicted positive"],
    )
    cm_df.to_csv(REPORTS_DIR / "confusion_matrix_cv.csv")

    report = classification_report(
        y_historical,
        y_pred_cv,
        target_names=["Biopsy negative", "Biopsy positive"],
        digits=3,
    )

    with open(REPORTS_DIR / "classification_report_cv.txt", "w") as f:
        f.write(report)

    print("\nCross-validated confusion matrix:")
    print(cm_df)

    print("\nCross-validated classification report:")
    print(report)

    # Fit final model on all historical data
    pipeline.fit(X_historical, y_historical)

    # Save fitted model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved final model to: {MODEL_PATH}")

    # Save selected feature coefficients
    all_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    selected_mask = pipeline.named_steps["selector"].get_support()
    selected_feature_names = all_feature_names[selected_mask]
    coefficients = pipeline.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": selected_feature_names,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(REPORTS_DIR / "selected_feature_coefficients.csv", index=False)

    print("\nSelected feature coefficients:")
    print(coef_df)


if __name__ == "__main__":
    main()