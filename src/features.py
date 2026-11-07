"""
Shared data loading, feature engineering and model-building code.

Both the training script (src/train.py), the improved notebook
(notebooks/02_improved_model.ipynb) and the Streamlit app (app.py) import
from this module, so the exact same transformations are applied during
training and at prediction time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "AIML Dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_pipeline.pkl"

# --------------------------------------------------------------------------
# Column definitions
# --------------------------------------------------------------------------
TARGET = "isFraud"

# Raw columns the model expects as input (what a user / upstream system sends).
# nameOrig / nameDest are dropped: they are near-unique IDs and do not generalize.
# isFlaggedFraud is dropped: it is the output of an existing rule-based system.
INPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_FEATURES = ["type"]

# Original numeric columns + the engineered features created in add_features().
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "hour",
]

# Transaction types in the dataset. Fraud only ever happens in TRANSFER and CASH_OUT.
TRANSACTION_TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]


def load_data(path=DATA_PATH):
    """Load only the columns we need, with compact dtypes (~0.3 GB instead of ~1.5 GB)."""
    return pd.read_csv(
        path,
        usecols=INPUT_COLUMNS + [TARGET],
        dtype={"type": "category", TARGET: "int8"},
    )


def add_features(df):
    """
    Add domain-driven features on top of the raw transaction columns.

    errorBalanceOrig / errorBalanceDest
        In a consistent ledger: new_balance = old_balance -/+ amount.
        These columns measure how far a transaction is from that rule.
        Fraudulent transactions very often empty the sender's account while the
        receiver's balance does not move accordingly, so these errors are
        strongly non-zero for fraud and close to zero for normal traffic.

    hour
        `step` is the hour since the start of the simulation (1 step = 1 hour).
        step % 24 gives the hour of day. Legitimate activity drops at night
        while fraud keeps the same rate, which makes the hour informative.
    """
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["hour"] = df["step"] % 24
    return df


def build_pipeline(random_state=42, n_jobs=-1):
    """
    Full model: feature engineering -> encoding -> gradient boosted trees.

    Keeping everything inside one sklearn Pipeline means the saved .pkl file
    takes raw transaction columns and handles all preprocessing itself.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            # Trees do not need scaling, so numeric columns pass through unchanged.
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",  # histogram-based training: fast on millions of rows
        eval_metric="aucpr",  # PR-AUC is the meaningful metric for rare-event problems
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features)),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def threshold_for_recall(y_true, y_proba, min_recall=0.95):
    """
    Return the highest probability threshold that still catches at least
    `min_recall` of the frauds (used on the validation set).

    The higher the threshold, the fewer false alarms (= higher precision), so
    this gives the best precision we can get without missing more fraud than
    we are willing to accept. The tutorial model had recall 0.94, so 0.95 is a
    "catch at least as much fraud as before" constraint.
    """
    from sklearn.metrics import precision_recall_curve

    _, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # recall[i] corresponds to thresholds[i]; recall decreases as the threshold grows.
    ok = np.where(recall[:-1] >= min_recall)[0]
    return float(thresholds[ok[-1]])
