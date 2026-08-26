"""
build_features.py

Builds the flood-risk feature/label table for Stage 2 (XGBoost classifier)
of the Flood_and_Water_Security_Toolkit.

Label: binary flood day, defined as discharge_spec exceeding the 95th
percentile of that catchment's TRAINING PERIOD discharge (not the full
record) -- avoiding leakage of test-period distribution into the label
definition itself, same principle as Stage 1's train-only normalization.

Features (all independent of the discharge record used for the label):
    - Antecedent precipitation: 3/7/30-day rolling sums
    - PET (same day)
    - Soil-moisture proxy: reused from Stage 1's calibrated conceptual
      baseline (simulate_discharge with return_states=True), NOT
      recalibrated here
    - Static catchment attributes: topographic, landcover, soil -- with
      hydrologic_attributes.csv deliberately excluded, since those columns
      are computed from the full discharge record and would leak into
      the label

Usage:
    python src/stage2_risk_classifier/build_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse Stage 1's data loading and calibrated baseline rather than
# duplicating either -- add stage1_lstm_forecast to the import path.
STAGE1_DIR = Path(__file__).resolve().parents[1] / "stage1_lstm_forecast"
sys.path.insert(0, str(STAGE1_DIR))

from data_utils import load_catchment_timeseries, CATCHMENT_AREA_KM2, TRAIN_END, VAL_END, split_train_val_test
from conceptual_baseline import simulate_discharge, load_checkpoint, PARAM_NAMES

PROJECT_ROOT = Path("/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit")
CAMELS_ATTR_DIR = PROJECT_ROOT / "data" / "camels_gb" / "camels_gb_v1_full" / "data"
FEATURES_DIR = PROJECT_ROOT / "data" / "stage2_features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_PERCENTILE = 95

# Static attribute columns -- deliberately excludes hydrologic_attributes.csv
# (leakage risk, see module docstring) and humaninfluence_attributes.csv
# (kept for the writeup narrative, not used as a model feature).
STATIC_ATTR_FILES = {
    "topographic": ["area", "dpsbar", "elev_mean"],
    "landcover": ["dwood_perc", "grass_perc", "crop_perc", "urban_perc"],
    "soil": ["sand_perc", "clay_perc", "porosity_cosby", "conductivity_cosby"],
}


def load_static_attributes(station_id: str) -> dict:
    """
    Load the selected static attribute columns for one station from the
    CAMELS-GB topographic, landcover, and soil attribute tables.
    """
    attrs = {}
    for category, columns in STATIC_ATTR_FILES.items():
        file_path = CAMELS_ATTR_DIR / f"CAMELS_GB_{category}_attributes.csv"
        df = pd.read_csv(file_path)
        row = df[df["gauge_id"] == int(station_id)]
        if row.empty:
            raise ValueError(f"Station {station_id} not found in {file_path.name}")
        for col in columns:
            attrs[col] = row.iloc[0][col]
    return attrs


def compute_soil_moisture_proxy(full_df: pd.DataFrame, station_id: str) -> np.ndarray:
    """
    Run the Stage 1 calibrated conceptual baseline forward across the FULL
    timeseries (not just train) using its saved parameters, returning the
    daily soil-moisture store trajectory. Parameters are loaded from
    Stage 1's checkpoint, not recalibrated here.
    """
    params = load_checkpoint(station_id)
    param_values = [params[name] for name in PARAM_NAMES]

    precip_mm = full_df["precipitation"].to_numpy()
    pet_mm = full_df["pet"].to_numpy()

    _, soil_store_mm = simulate_discharge(precip_mm, pet_mm, param_values, return_states=True)
    return soil_store_mm


def compute_antecedent_precip(full_df: pd.DataFrame) -> pd.DataFrame:
    """Add 3/7/30-day rolling precipitation sums as columns."""
    df = full_df.copy()
    df["precip_3d"] = df["precipitation"].rolling(window=3, min_periods=3).sum()
    df["precip_7d"] = df["precipitation"].rolling(window=7, min_periods=7).sum()
    df["precip_30d"] = df["precipitation"].rolling(window=30, min_periods=30).sum()
    return df


def compute_flood_label(full_df: pd.DataFrame, train_end: str) -> tuple[pd.Series, float]:
    """
    Binary flood-day label: discharge_spec exceeding the FLOOD_PERCENTILE
    threshold of the TRAINING PERIOD's discharge distribution. The
    threshold is computed once from train data only, then applied across
    the full series -- so val/test labels don't leak their own
    distribution into the threshold definition.
    """
    train_discharge = full_df.loc[:train_end, "discharge_spec"]
    threshold = np.nanpercentile(train_discharge, FLOOD_PERCENTILE)
    label = (full_df["discharge_spec"] > threshold).astype(int)
    label[full_df["discharge_spec"].isna()] = np.nan  # can't label a day with no discharge
    return label, threshold


def build_feature_table(station_id: str) -> pd.DataFrame:
    """
    Build the complete feature/label table for one catchment: antecedent
    precipitation, PET, soil-moisture proxy, static attributes, and the
    flood-day label. Rows with any NaN (rolling-window warmup, or missing
    discharge for the label) are dropped.
    """
    full_df = load_catchment_timeseries(station_id)
    full_df = compute_antecedent_precip(full_df)

    full_df["soil_moisture_proxy_mm"] = compute_soil_moisture_proxy(full_df, station_id)

    label, threshold = compute_flood_label(full_df, TRAIN_END)
    full_df["flood_day"] = label

    static_attrs = load_static_attributes(station_id)
    for col, value in static_attrs.items():
        full_df[col] = value

    feature_cols = (
        ["precip_3d", "precip_7d", "precip_30d", "pet", "soil_moisture_proxy_mm"]
        + list(static_attrs.keys())
    )
    keep_cols = feature_cols + ["flood_day"]

    table = full_df[keep_cols].copy()
    n_before = len(table)
    table = table.dropna()
    n_after = len(table)

    print(f"  Flood threshold (train-period {FLOOD_PERCENTILE}th pct discharge_spec): {threshold:.3f} mm/day")
    print(f"  Rows: {n_before} -> {n_after} after dropping NaN (rolling-window warmup + missing discharge)")
    print(f"  Flood day rate: {table['flood_day'].mean():.3%}")

    return table


if __name__ == "__main__":
    for station_id in CATCHMENT_AREA_KM2:
        print(f"Building feature table for station {station_id}...")
        table = build_feature_table(station_id)

        output_path = FEATURES_DIR / f"features_{station_id}.csv"
        table.to_csv(output_path)
        print(f"  Saved: {output_path}")
        print()