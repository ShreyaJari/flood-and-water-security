"""
xgboost_classifier.py

XGBoost flood-day classifier for Stage 2 of the Flood_and_Water_Security_Toolkit.
Trained independently per catchment on the feature/label tables built by
build_features.py. Uses fixed, sensible-default hyperparameters rather than
a search -- consistent with Stage 1's fixed-bounds approach, and avoids
overfitting hyperparameters to a validation set with a small positive class.

Evaluation uses precision/recall/F1 on the flood class and a
precision-recall curve, NOT accuracy or ROC-AUC -- with ~5% positive rate,
accuracy is misleading (always-negative would score ~94-95%) and ROC can
look deceptively good under class imbalance.

Usage:
    python src/stage2_risk_classifier/xgboost_classifier.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)

PROJECT_ROOT = Path("/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit")
FEATURES_DIR = PROJECT_ROOT / "data" / "stage2_features"
MODEL_DIR = PROJECT_ROOT / "models" / "stage2_xgboost"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STATION_IDS = ["39002", "76007"]

TRAIN_END = "1997-09-30"
VAL_END = "2005-09-30"

# Fixed hyperparameters -- reasonable defaults for a small tabular dataset,
# not tuned via search. max_depth kept shallow given the limited number of
# features and positive examples, to reduce overfitting risk.
XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",  # area under precision-recall curve, appropriate for imbalance
    "random_state": 42,
}


def load_split(station_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a station's feature table and split into train/val/test by date."""
    df = pd.read_csv(FEATURES_DIR / f"features_{station_id}.csv", index_col=0, parse_dates=True)
    train = df.loc[:TRAIN_END]
    val = df.loc[TRAIN_END:VAL_END].iloc[1:]
    test = df.loc[VAL_END:].iloc[1:]
    return train, val, test


def train_classifier(station_id: str) -> dict:
    """
    Train an XGBoost flood-day classifier for one catchment, evaluate on
    val/test using precision/recall/F1 and average precision, and save a
    checkpoint (model + feature columns + threshold metadata).
    """
    train, val, test = load_split(station_id)
    feature_cols = [c for c in train.columns if c != "flood_day"]

    x_train, y_train = train[feature_cols], train["flood_day"]
    x_val, y_val = val[feature_cols], val["flood_day"]
    x_test, y_test = test[feature_cols], test["flood_day"]

    # scale_pos_weight balances the ~5% positive class rather than letting
    # the model default toward always predicting "no flood"
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict(x_val)
    val_proba = model.predict_proba(x_val)[:, 1]
    test_pred = model.predict(x_test)
    test_proba = model.predict_proba(x_test)[:, 1]

    val_scores = {
        "precision": precision_score(y_val, val_pred),
        "recall": recall_score(y_val, val_pred),
        "f1": f1_score(y_val, val_pred),
        "avg_precision": average_precision_score(y_val, val_proba),
    }
    test_scores = {
        "precision": precision_score(y_test, test_pred),
        "recall": recall_score(y_test, test_pred),
        "f1": f1_score(y_test, test_pred),
        "avg_precision": average_precision_score(y_test, test_proba),
    }

    checkpoint_path = MODEL_DIR / f"xgb_{station_id}.json"
    model.save_model(checkpoint_path)

    metadata_path = MODEL_DIR / f"xgb_{station_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "scale_pos_weight": float(scale_pos_weight),
                "n_train_positive": int(n_pos),
                "n_train_total": int(len(y_train)),
                "val_scores": val_scores,
                "test_scores": test_scores,
            },
            f,
            indent=2,
        )

    print(f"  Train positive rate: {n_pos}/{len(y_train)} ({n_pos/len(y_train):.3%}), scale_pos_weight={scale_pos_weight:.2f}")
    print(f"  Val:  precision={val_scores['precision']:.3f}, recall={val_scores['recall']:.3f}, "
          f"f1={val_scores['f1']:.3f}, avg_precision={val_scores['avg_precision']:.3f}")
    print(f"  Test: precision={test_scores['precision']:.3f}, recall={test_scores['recall']:.3f}, "
          f"f1={test_scores['f1']:.3f}, avg_precision={test_scores['avg_precision']:.3f}")
    print(f"  Saved: {checkpoint_path}")
    print(f"  Saved: {metadata_path}")

    return {"model": model, "test_scores": test_scores, "feature_cols": feature_cols}


if __name__ == "__main__":
    for station_id in STATION_IDS:
        print(f"Training XGBoost classifier for station {station_id}...")
        train_classifier(station_id)
        print()