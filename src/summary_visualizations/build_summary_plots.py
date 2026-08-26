"""
build_summary_plots.py

Builds four summary visualizations tying together Stage 1 and Stage 2
results across both catchments, reusing saved checkpoints -- no
retraining, no recalibration:

    1. Precision-recall curves (Stage 2 XGBoost, Thames vs. Eden)
    2. SHAP top-5 feature ranking, side by side (Thames vs. Eden)
    3. NSE/KGE grouped bar chart: baseline vs. LSTM, both catchments
    4. Topographic comparison (dpsbar, elev_mean) explaining the SHAP result

Figures saved to results/summary_visualizations/.

Usage:
    python src/summary_visualizations/build_summary_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, average_precision_score

PROJECT_ROOT = Path("/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit")
sys.path.insert(0, str(PROJECT_ROOT / "src" / "stage1_lstm_forecast"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "stage2_risk_classifier"))

from conceptual_baseline import evaluate as baseline_evaluate, load_checkpoint as load_baseline_checkpoint
from data_utils import load_catchment_split, CATCHMENT_AREA_KM2

RESULTS_DIR = PROJECT_ROOT / "results" / "summary_visualizations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = PROJECT_ROOT / "data" / "stage2_features"
XGB_MODEL_DIR = PROJECT_ROOT / "models" / "stage2_xgboost"
LSTM_MODEL_DIR = PROJECT_ROOT / "models" / "stage1_lstm"
CAMELS_ATTR_DIR = PROJECT_ROOT / "data" / "camels_gb" / "camels_gb_v1_full" / "data"

STATION_IDS = ["39002", "76007"]
STATION_LABELS = {"39002": "Thames at Days Weir", "76007": "Eden at Sheepmount"}
STATION_COLORS = {"39002": "tab:blue", "76007": "tab:orange"}

VAL_END = "2005-09-30"


# ---------------------------------------------------------------------------
# 1. Precision-recall curves
# ---------------------------------------------------------------------------

def plot_precision_recall_curves() -> None:
    """Overlay PR curves for both catchments' Stage 2 classifiers on the test period."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for station_id in STATION_IDS:
        model = xgb.XGBClassifier()
        model.load_model(XGB_MODEL_DIR / f"xgb_{station_id}.json")

        with open(XGB_MODEL_DIR / f"xgb_{station_id}_metadata.json") as f:
            metadata = json.load(f)
        feature_cols = metadata["feature_cols"]

        df = pd.read_csv(FEATURES_DIR / f"features_{station_id}.csv", index_col=0, parse_dates=True)
        test = df.loc[VAL_END:].iloc[1:]
        x_test, y_test = test[feature_cols], test["flood_day"]

        proba = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)

        label = STATION_LABELS[station_id]
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})", color=STATION_COLORS[station_id], linewidth=2)

    baseline_rate = 0.05  # approximate flood-day base rate, both catchments
    ax.axhline(baseline_rate, color="gray", linestyle="--", linewidth=1, label=f"No-skill baseline (~{baseline_rate:.0%})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Stage 2: Precision-Recall Curves (Test Period)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    output_path = RESULTS_DIR / "precision_recall_curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# 2. SHAP top-5 feature ranking, side by side
# ---------------------------------------------------------------------------

def plot_shap_comparison() -> None:
    """Compute SHAP values for both catchments and plot top-5 features side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

    for ax, station_id in zip(axes, STATION_IDS):
        model = xgb.XGBClassifier()
        model.load_model(XGB_MODEL_DIR / f"xgb_{station_id}.json")

        with open(XGB_MODEL_DIR / f"xgb_{station_id}_metadata.json") as f:
            metadata = json.load(f)
        feature_cols = metadata["feature_cols"]

        df = pd.read_csv(FEATURES_DIR / f"features_{station_id}.csv", index_col=0, parse_dates=True)
        test = df.loc[VAL_END:].iloc[1:]
        x_test = test[feature_cols]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)

        mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=True).tail(5)

        ax.barh(mean_abs_shap.index, mean_abs_shap.values, color=STATION_COLORS[station_id])
        ax.set_title(STATION_LABELS[station_id])
        ax.set_xlabel("Mean |SHAP value|")

    fig.suptitle("Stage 2: Top-5 Feature Importance by Catchment", fontsize=13)
    fig.tight_layout()

    output_path = RESULTS_DIR / "shap_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# 3. NSE/KGE grouped bar chart: baseline vs. LSTM
# ---------------------------------------------------------------------------

def plot_nse_kge_comparison() -> None:
    """Grouped bar chart: baseline vs. LSTM test NSE/KGE, both catchments."""
    records = []

    for station_id in STATION_IDS:
        split = load_catchment_split(station_id)
        baseline_params = load_baseline_checkpoint(station_id)
        baseline_scores = baseline_evaluate(split.test, baseline_params)

        with open(LSTM_MODEL_DIR / f"lstm_{station_id}.pt", "rb"):
            pass  # existence check only; actual scores are in the checkpoint dict
        import torch
        checkpoint = torch.load(LSTM_MODEL_DIR / f"lstm_{station_id}.pt", map_location="cpu", weights_only=False)

        records.append({"station": STATION_LABELS[station_id], "model": "Conceptual baseline", "NSE": baseline_scores["nse"], "KGE": baseline_scores["kge"]})
        records.append({"station": STATION_LABELS[station_id], "model": "LSTM", "NSE": checkpoint["test_nse"], "KGE": checkpoint["test_kge"]})

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric in zip(axes, ["NSE", "KGE"]):
        pivot = df.pivot(index="station", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0, color=["#7f8fa6", "#e17055"])
        ax.set_title(f"Test {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.legend(title="")

    fig.suptitle("Stage 1: Conceptual Baseline vs. LSTM (Test Period)", fontsize=13)
    fig.tight_layout()

    output_path = RESULTS_DIR / "nse_kge_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# 4. Topographic comparison (explains the SHAP result)
# ---------------------------------------------------------------------------

def plot_topographic_comparison() -> None:
    """Bar chart of dpsbar (mean drainage path slope) and elev_mean, both catchments."""
    topo_df = pd.read_csv(CAMELS_ATTR_DIR / "CAMELS_GB_topographic_attributes.csv")

    rows = topo_df[topo_df["gauge_id"].isin([int(sid) for sid in STATION_IDS])].copy()
    rows["label"] = rows["gauge_id"].astype(str).map(STATION_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].bar(rows["label"], rows["dpsbar"], color=[STATION_COLORS[str(g)] for g in rows["gauge_id"]])
    axes[0].set_title("Mean Drainage Path Slope (dpsbar)")
    axes[0].set_ylabel("m/km")

    axes[1].bar(rows["label"], rows["elev_mean"], color=[STATION_COLORS[str(g)] for g in rows["gauge_id"]])
    axes[1].set_title("Mean Elevation")
    axes[1].set_ylabel("m")

    fig.suptitle("Topographic Basis for Fast (Eden) vs. Slow (Thames) Response", fontsize=12)
    fig.tight_layout()

    output_path = RESULTS_DIR / "topographic_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    print("1. Precision-recall curves...")
    plot_precision_recall_curves()

    print("2. SHAP feature comparison...")
    plot_shap_comparison()

    print("3. NSE/KGE comparison...")
    plot_nse_kge_comparison()

    print("4. Topographic comparison...")
    plot_topographic_comparison()