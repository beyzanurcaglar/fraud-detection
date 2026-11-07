"""
Reproducible training script for the improved fraud detection model.

Usage (from the project root):
    python -m src.train

Steps:
    1. Load the PaySim-based dataset (data/AIML Dataset.csv)
    2. Split into train / validation / test (stratified, same 70/30 test split as the tutorial)
    3. Train the XGBoost pipeline defined in src/features.py
    4. Pick a decision threshold on the validation set (best precision at >= 95% recall)
    5. Evaluate on the untouched test set and save model + threshold to models/
"""

import json

import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import (
    INPUT_COLUMNS,
    MODEL_PATH,
    PROJECT_ROOT,
    TARGET,
    threshold_for_recall,
    build_pipeline,
    load_data,
)

RANDOM_STATE = 42


def main():
    # 1. Load data
    df = load_data()
    X, y = df[INPUT_COLUMNS], df[TARGET]
    print(f"Loaded {len(df):,} transactions, fraud rate = {y.mean():.4%}")

    # 2. Split. The 30% test split is identical to the tutorial so results are comparable.
    #    A further 15% of the training data is held out to tune the decision threshold.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )

    # 3. Train
    pipeline = build_pipeline(random_state=RANDOM_STATE)
    pipeline.fit(X_fit, y_fit)

    # 4. Threshold tuning on validation data (never on the test set):
    #    highest threshold that still keeps recall >= 95%  ->  best precision at that recall
    threshold = threshold_for_recall(y_val, pipeline.predict_proba(X_val)[:, 1], min_recall=0.95)
    print(f"Chosen decision threshold: {threshold:.3f}")

    # 5. Final evaluation on the test set
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    print(classification_report(y_test, y_pred, digits=4))
    print(confusion_matrix(y_test, y_pred))

    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    print(json.dumps(metrics, indent=2))

    # Save the pipeline together with its threshold so the app uses the same cut-off.
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, MODEL_PATH)
    (PROJECT_ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
