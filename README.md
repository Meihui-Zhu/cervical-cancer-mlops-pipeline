# Cervical Cancer MLOps Pipeline

This repository contains an end-to-end machine learning pipeline for cervical cancer biopsy risk prediction using the UCI Cervical Cancer (Risk Factors) dataset.

## Project overview

The project simulates a clinical screening workflow in which new patient records arrive as daily batches.


Because the original UCI dataset is static, the pipeline first splits the data into:

- historical records used for model development and baseline distribution monitoring;
- simulated future records assigned synthetic arrival dates to represent daily incoming batches.

The prediction task is:

> Predict whether a record has a positive biopsy result using given features. 

The target variable is:

```text
Biopsy
```
. 
Other target variables in the dataset, including `Hinselmann`, `Schiller`, and `Citology`, are excluded from model features to avoid target leakage.


## Data source

This project uses a local copy of the Cervical Cancer (Risk Factors) dataset from the UCI Machine Learning Repository.

- Dataset: Cervical Cancer (Risk Factors)
- Repository: UCI Machine Learning Repository
- DOI: 10.24432/C5Z310
- License: Creative Commons Attribution 4.0 International License (CC BY 4.0)


The original raw data file is stored under:

```text
data/external/
```

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
└── README.md
```


## Setup

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

1. clean generated outputs from previous runs;
2. load the original UCI dataset;
3. create a 70/30 stratified historical/future split using `Biopsy`;
4. assign synthetic daily arrival dates to the future records;
5. train and evaluate the final model on historical data;
6. save the trained model;
7. run inference for all simulated daily batches;
8. save predictions to CSV files and SQLite;
9. run drift monitoring for each daily batch.


## Run individual steps

### 1. Train the model

```bash
python3 scripts/run_training.py
```

This generates:

```text
models/biopsy_sparse_logistic_pipeline.joblib
reports/cv_results_sparse_logistic.csv
reports/confusion_matrix_cv.csv
reports/classification_report_cv.txt
reports/selected_feature_coefficients.csv
reports/historical_cv_prediction_scores.csv
data/processed/historical_data.csv
data/processed/future_data_with_batches.csv
```


### 2. Run inference for one daily batch

```bash
python3 scripts/run_daily_inference.py --date 2026-05-08
```


### 3. Run inference for all daily batches

```bash
python3 scripts/run_daily_inference.py --date all
```

Prediction outputs are saved under:

```text
data/predictions/
```

Predictions are also written to the SQLite database:

```text
data/pipeline.db
```


### 4. Run drift checks for one daily batch
```bash
python3 scripts/run_drift_check.py --date 2026-05-08
```


### 5. Run drift checks for all daily batches

```bash
python3 scripts/run_drift_check.py --date all
```

Drift reports are saved under:

```text
reports/
```

### 6. Run optional drift stress test

```bash
python3 scripts/run_drift_stress_test.py
```
The stress test artificially increases missingness in 6 variables:
`Hormonal Contraceptives`, `Hormonal Contraceptives (years)`, `IUD`, `IUD (years)`, `STDs`, `STDs (number)`, to simulate the situation where several questions in the survey questionnaire fail. To verify that the monitoring module can flag a data-quality shift. This is only a synthetic test and is not treated as real incoming data.



## Model pipeline
tbc


