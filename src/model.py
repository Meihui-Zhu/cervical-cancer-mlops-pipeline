# src/model.py

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


NUMERIC_FEATURES = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives (years)",
    "IUD (years)",
    "STDs (number)",
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
    "STDs: Number of diagnosis",
]


BINARY_FEATURES = [
    "Smokes",
    "Hormonal Contraceptives",
    "IUD",
    "STDs",
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
]


def build_preprocessor():
    """Build preprocessing pipeline for numeric and binary features."""

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def build_sparse_logistic_pipeline():
    """
    Build the final model pipeline.

    L1 logistic regression is used for feature selection.
    L2 logistic regression is used as the final classifier.
    """

    preprocessor = build_preprocessor()

    feature_selector = SelectFromModel(
        estimator=LogisticRegression(
            # L1 regularization
            l1_ratio=1.0,
            solver="liblinear",
            class_weight="balanced",
            C=0.5,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        max_features=10,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("selector", feature_selector),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    return pipeline