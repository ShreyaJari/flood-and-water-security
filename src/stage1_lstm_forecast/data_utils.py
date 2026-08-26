"""
data_utils.py

Shared data loading and splitting for Stage 1 (LSTM forecasting) of the
Flood_and_Water_Security_Toolkit. Used by both conceptual_baseline.py and
lstm_model.py so the train/validation/test split is identical across both
models -- a fair benchmark depends on that.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(
    "/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit"
    "/data/camels_gb/camels_gb_v1_full/data/timeseries"
)

# Confirmed via NRFA API in Stage 1 verification.
CATCHMENT_AREA_KM2 = {
    "39002": 3444.7,  # Thames at Days Weir
    "76007": 2286.5,  # Eden at Sheepmount
}

TRAIN_END = "1997-09-30"
VAL_END = "2005-09-30"
# Test period runs to the end of the record (2015-09-30).


@dataclass
class CatchmentSplit:
    station_id: str
    catchment_area_km2: float
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_catchment_timeseries(station_id: str) -> pd.DataFrame:
    """
    Load the raw CAMELS-GB hydromet timeseries CSV for a single station id
    and parse the date column.
    """
    matches = list(DATA_ROOT.glob(f"*_{station_id}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No timeseries file found for station {station_id} in {DATA_ROOT}")
    if len(matches) > 1:
        raise ValueError(f"Multiple timeseries files matched station {station_id}: {matches}")

    df = pd.read_csv(matches[0], parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def split_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a catchment timeseries into train / validation / test periods
    using fixed dates shared across both catchments, matching the
    convention used in Lane et al. (2019) GB-wide benchmarking.
    """
    train = df.loc[:TRAIN_END]
    val = df.loc[TRAIN_END:VAL_END].iloc[1:]  # avoid double-counting the boundary day
    test = df.loc[VAL_END:].iloc[1:]
    return train, val, test


def load_catchment_split(station_id: str) -> CatchmentSplit:
    """
    Convenience wrapper: load a catchment's timeseries and return it
    already split into train/val/test, alongside its catchment area.
    """
    df = load_catchment_timeseries(station_id)
    train, val, test = split_train_val_test(df)
    return CatchmentSplit(
        station_id=station_id,
        catchment_area_km2=CATCHMENT_AREA_KM2[station_id],
        train=train,
        val=val,
        test=test,
    )


def mmday_to_cumecs(q_mmday: pd.Series, catchment_area_km2: float) -> pd.Series:
    """
    Convert specific discharge (mm/day) to volumetric discharge (m3/s),
    given catchment area in km2.

    1 mm/day over 1 km2 = 1000 m3/day = 1000 / 86400 m3/s.
    """
    return q_mmday * catchment_area_km2 * 1000 / 86400


def verify_unit_conversion(df: pd.DataFrame, catchment_area_km2: float, tol: float = 0.05) -> None:
    """
    Sanity check: recompute discharge_vol from discharge_spec using
    mmday_to_cumecs() and compare against the actual discharge_vol column.
    This confirms the conversion formula before we trust it on simulated
    (not observed) discharge later.
    """
    recomputed = mmday_to_cumecs(df["discharge_spec"], catchment_area_km2)
    relative_error = ((recomputed - df["discharge_vol"]).abs() / df["discharge_vol"].replace(0, pd.NA)).mean()
    print(f"  Mean relative error, recomputed vs. actual discharge_vol: {relative_error:.4f}")
    if relative_error > tol:
        print(f"  WARNING: relative error exceeds tolerance ({tol}). Check unit conversion.")
    else:
        print("  Unit conversion verified within tolerance.")


if __name__ == "__main__":
    for station_id in CATCHMENT_AREA_KM2:
        print(f"Loading station {station_id}...")
        split = load_catchment_split(station_id)
        print(f"  train: {len(split.train)} days ({split.train.index.min()} to {split.train.index.max()})")
        print(f"  val:   {len(split.val)} days ({split.val.index.min()} to {split.val.index.max()})")
        print(f"  test:  {len(split.test)} days ({split.test.index.min()} to {split.test.index.max()})")
        verify_unit_conversion(split.train, split.catchment_area_km2)
        print()