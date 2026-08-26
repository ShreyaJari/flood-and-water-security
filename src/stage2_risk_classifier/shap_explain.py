"""
shap_explain.py

SHAP explainability for Stage 2's XGBoost flood-risk classifiers. Loads
the saved model checkpoints (no retraining) and computes SHAP values on
the test-period features for both catchments, producing summary plots
that show which features drive each catchment's flood predictions.

This is the piece that delivers the "explainable" half of the brief --
and lets us check whether Eden's risk is genuinely more
precipitation/soil-moisture driven than Thames', as the physical story
from Stages 1-2 would predict, rather than just asserting it.

Usage:
    python src/stage2_risk_classifier/shap_explain.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb

PROJECT_ROOT = Path("/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit")
FEATURES_DIR = PROJECT_ROOT / "data" / "stage2_features"
MODEL_DIR = PROJECT_ROOT / "models" / "stage2_xgboost"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage2_risk_classifier"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATION_IDS = ["39002", "76007"]
STATION_LABELS = {
    "39002": "Thames at Days Weir",
    "76007": "Eden at Sheepmount",
}

VAL_END = "2005-09-30"


def load_model_and_metadata(station_id: str) -> tuple[xgb.XGBClassifier, dict]:
    """Load a saved XGBoost checkpoint and its metadata (feature columns etc.)."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / f"xgb_{station_id}.json")

    with open(MODEL_DIR / f"xgb_{station_id}_metadata.json") as f:
        metadata = json.load(f)

    return model, metadata


def load_test_features(station_id: str, feature_cols: list[str]) -> pd.DataFrame:
    """Load the test-period feature rows for a station, same split as xgboost_classifier.py."""
    df = pd.read_csv(FEATURES_DIR / f"features_{station_id}.csv", index_col=0, parse_dates=True)
    test = df.loc[VAL_END:].iloc[1:]
    return test[feature_cols]


def plot_shap_summary(station_id: str) -> None:
    """Compute SHAP values on the test set and save a summary beeswarm plot."""
    label = STATION_LABELS[station_id]
    model, metadata = load_model_and_metadata(station_id)
    feature_cols = metadata["feature_cols"]

    x_test = load_test_features(station_id, feature_cols)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    plt.figure()
    shap.summary_plot(shap_values, x_test, show=False)
    plt.title(f"{label} ({station_id}) -- SHAP feature importance")
    plt.tight_layout()

    output_path = RESULTS_DIR / f"shap_summary_{station_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    # Mean absolute SHAP value per feature -- a compact numeric ranking
    # alongside the visual summary, useful for the writeup table.
    mean_abs_shap = pd.Series(
        abs(shap_values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)
    print(f"  Top features by mean |SHAP value|:")
    for feature, value in mean_abs_shap.head(5).items():
        print(f"    {feature}: {value:.4f}")


if __name__ == "__main__":
    for station_id in STATION_IDS:
        print(f"Computing SHAP values for station {station_id}...")
        plot_shap_summary(station_id)
        print()