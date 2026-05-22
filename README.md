# Cervical Cancer MLOps Pipeline

This repository contains an end-to-end machine learning pipeline for cervical cancer biopsy risk prediction using the UCI Cervical Cancer (Risk Factors) dataset.

## Project overview

The project simulates a clinical screening workflow in which new patient records arrive as daily batches.


Because the original UCI dataset is static, the pipeline first splits the data into:

- historical records used for model development and baseline distribution monitoring;
- simulated future records assigned synthetic arrival dates to represent daily incoming batches.

The prediction task is:

> Predict whether a record has a positive biopsy result using given features. 

The target variable is:`Biopsy`. 

Other target variables in the dataset, including `Hinselmann`, `Schiller`, and `Citology`, are excluded from model features to avoid target leakage.


## Data source

This project uses a local copy of the Cervical Cancer (Risk Factors) dataset from the UCI Machine Learning Repository.

- Dataset: Cervical Cancer (Risk Factors)
- Repository: UCI Machine Learning Repository
- DOI: 10.24432/C5Z310
- License: Creative Commons Attribution 4.0 International License (CC BY 4.0)


The original raw data file is stored under: `data/external/`

The dataset is included locally so that the full pipeline can be run without requiring network access.


## Repository structure

```text
cervical-cancer-mlops-pipeline/
├── data/
│   ├── external/                # Original raw UCI dataset
│   ├── processed/               # Generated historical/future split files
│   └── predictions/             # Generated daily prediction CSV files
│   └── pipeline.db              # SQLite Database for data ingesting log and prediction results
├── models/                      # Saved trained model
├── reports/                     # Evaluation, model inspection, and drift reports
├── scripts/
│   ├── run_training.py           # Train model and generate evaluation reports
│   ├── run_daily_inference.py    # Run inference for one or all daily batches
│   ├── run_drift_check.py        # Run drift checks for one or all daily batches
│   ├── run_drift_stress_test.py  # Optional synthetic drift stress test
│   └── run_full_demo.py          # Run the full end-to-end pipeline
├── src/
│   ├── config.py                 # Project paths and constants
│   ├── data.py                   # Data loading, splitting, and batch simulation
│   ├── model.py                  # Preprocessing and model pipeline
│   ├── drift.py                  # Drift monitoring functions
│   └── storage.py                # SQLite logging and prediction storage
├── requirements.txt
├── report.pdf                    # Detailed report explaining pipeline design and engineering 
└── README.md
```


## Setup

Clone the repository and move to the project root directory:

```bash
git clone https://github.com/Meihui-Zhu/cervical-cancer-mlops-pipeline
cd cervical-cancer-mlops-pipeline
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the full demo

To run the complete end-to-end pipeline:
```bash
python3 scripts/run_full_demo.py
```

This command will:

1. Clean generated outputs from previous runs;
2. Load the original UCI dataset;
3. Create a 70/30 stratified historical/future split using `Biopsy`;
4. Assign synthetic daily arrival dates (from `2026-05-08` to `2026-05-21`) to the future records;
5. Train and evaluate the final model on historical data;
6. Save the trained model;
7. Run inference for all simulated daily batches;
8. Save predictions to CSV files and SQLite;
9. Run drift monitoring for each daily batch.


### Optional: run the drift stress test

The full demo runs drift monitoring on the normal simulated daily batches. In this stable scenario, no major drift is expected because the historical and future records come from the same source dataset.

To verify that the drift monitoring logic can detect a clear data-quality shift, run the optional synthetic stress test:

```bash
python3 scripts/run_drift_stress_test.py
```

To simulate the situation where several questions in the survey questionnaire fail, this script artificially increases missingness in 6 selected variables:
`Hormonal Contraceptives`, `Hormonal Contraceptives (years)`, `IUD`, `IUD (years)`, `STDs`, `STDs (number)`
 and saves the stress-test drift report under:

- `reports/drift_stress_test_missingness_2026-05-22.csv`

The stress test artificially is designed to verify that the monitoring module can flag a data-quality shift. This is only a synthetic test and is not treated as real incoming data.


## Run individual steps

### 1. Train the model

```bash
python3 scripts/run_training.py
```

This generates:

- models/biopsy_sparse_logistic_pipeline.joblib
- reports/cv_results_sparse_logistic.csv
- reports/confusion_matrix_cv.csv
- reports/classification_report_cv.txt
- reports/selected_feature_coefficients.csv
- reports/historical_cv_prediction_scores.csv
- data/processed/historical_data.csv
- data/processed/future_data_with_batches.csv



### 2. Run inference for one daily batch

```bash
python3 scripts/run_daily_inference.py --date 2026-05-08
```


### 3. Run inference for all daily batches

```bash
python3 scripts/run_daily_inference.py --date all
```

Prediction outputs are saved under: `data/predictions/`

Predictions are also written to the SQLite database: `data/pipeline.db`


### 4. Run drift checks for one daily batch
```bash
python3 scripts/run_drift_check.py --date 2026-05-08
```


### 5. Run drift checks for all daily batches

```bash
python3 scripts/run_drift_check.py --date all
```

Drift reports are saved under:

`reports/`

### 6. Run optional drift stress test

```bash
python3 scripts/run_drift_stress_test.py
```




## Model pipeline
The final model is a sparse logistic regression pipeline:


raw input features
1. Missing value imputation
2. Missingness indicators
3. Scaling for numeric features
4. L1-based feature selection
5. Class-weighted L2-based logistic regression classifier


The model uses:

- Median imputation for numeric features;
- Most-frequent imputation for binary features;
- Missingness indicators to preserve information about missing values;
- Class weighting to handle the rare positive biopsy outcome;
- L1-based feature selection to reduce redundancy among correlated features;
- A final logistic regression classifier for transparent baseline prediction.


Model development compared several candidate approaches during exploration, including full logistic regression, sparse logistic regression, XGBoost, and Random Forest. The sparse logistic regression pipeline was selected because it gave the best balance of cross-validation performance, stability, interpretability, and simplicity for this small imbalanced dataset.


## Storage design

This project uses both flat files and a lightweight SQLite database.

Flat files are used for:

- the original public dataset;
- generated historical/future split files;
- prediction CSV files;
- model;
- evaluation and drift reports.

SQLite is used for structured operational records that may need to be queried over time: `data/pipeline.db`.
Using a database here has several advantages over storing everything as separate CSV files:

- It allows efficient filtering and aggregation by batch date, record ID, and model version. For example, the system can quickly summarise how many records were processed or predicted as positive within a given time window.

- It better resembles how a hospital or screening system would store operational prediction records, where different users or downstream services may need to query the same records consistently.

- It makes it easier to update or link delayed outcome information later. For example, if a definitive biopsy result becomes available after the initial prediction, the stored prediction record can be updated with the new label for subsequent evaluation.


The database contains two tables:

- ingestion_log
- predictions

The `ingestion_log` table records which daily batches were processed, how many records they contained, and whether processing completed successfully. 

Example `ingestion_log` row:

| batch_date | n_records | n_positive_labels | positive_rate | status | created_at |
|---|---:|---:|---:|---|---|
| 2026-05-08 | 19 | 1 | 0.052632 | completed | 2026-05-21 21:31:12 |

The `predictions` table stores one row per model prediction, including the batch date, synthetic record ID, predicted biopsy-positive probability, predicted label, true label for offline evaluation, and model version.

Example `predictions` row:

| prediction_id | batch_date | record_id | biopsy_positive_probability | predicted_label | true_biopsy_label | model_version | created_at |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | 2026-05-08 | 137 | 0.642 | 1 | 0 | sparse_logistic_v1 | 2026-05-21 21:31:12 |


## Drift monitoring

The historical data is treated as the baseline distribution. Each incoming daily batch is compared against this baseline using:

1. missingness drift;
2. numeric feature mean-shift checks;
3. prediction score drift.

Missingness drift is flagged when the missing rate of a feature differs from the historical baseline by more than 20%.

Numeric feature drift is flagged when the daily batch mean differs from the historical mean by more than two historical standard deviations.

Prediction score drift compares daily predicted biopsy-positive probabilities against historical cross-validated prediction probabilities.

In the default simulated daily batches, no major drift is expected because the historical and future records are sampled from the same source dataset. The optional stress test demonstrates that the monitoring module can flag a synthetic data-quality shift.

## Notes on interpretation

This project is an engineering demonstration, not a clinically deployable model.

The dataset is small, static, and highly imbalanced. The biopsy-positive class is rare, so evaluation focuses on metrics such as ROC-AUC, average precision, recall, precision, F1-score, and confusion matrices rather than accuracy alone.

Selected features and model coefficients are used only for model inspection. They should not be interpreted as causal effects or clinically validated risk factors.

Diagnosis-related variables such as `Dx:HPV` and `Dx:CIN` are treated here as historical medical-record features. In a real screening deployment, their timing and availability would need to be verified carefully to avoid temporal leakage.

