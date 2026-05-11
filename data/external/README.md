# Cervical Cancer Risk Factors Dataset

This directory contains a local copy of the **Cervical Cancer (Risk Factors)** dataset from the UCI Machine Learning Repository.

The dataset is included in this repository to make the end-to-end pipeline fully reproducible without requiring network access during execution.

## Source

- Dataset: Cervical Cancer (Risk Factors)
- Repository: UCI Machine Learning Repository
- DOI: 10.24432/C5Z310
- Dataset page: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

## Dataset description

The dataset focuses on the prediction of indicators and diagnosis of cervical cancer. It contains demographic information, habits, and historical medical records for 858 patients. The dataset contains missing values because some patients did not answer certain questions due to privacy concerns.

## License and attribution

The dataset is provided by the UCI Machine Learning Repository and is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Please cite the dataset as:

Fernandes, K., Cardoso, J. S., & Fernandes, J. (2017). Cervical Cancer (Risk Factors). UCI Machine Learning Repository. https://doi.org/10.24432/C5Z310

## Use in this project

This project uses the dataset as a public, reproducible proxy for a clinical screening dataset.

Because the original dataset is static, the pipeline assigns synthetic arrival dates to the records and simulates daily screening batches. These simulated batches are used to demonstrate data ingestion, preprocessing, model training, inference, prediction storage, and drift monitoring in an end-to-end machine learning system.

The local data file should be treated as an immutable external data source. Pipeline-generated files are stored separately under `data/raw/`, `data/processed/`, and `data/predictions/`.