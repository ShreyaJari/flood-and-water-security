"""
conceptual_baseline.py

A simple 3-store linear-reservoir rainfall-runoff model, used as the
conceptual baseline benchmark against the LSTM in Stage 1 of the
Flood_and_Water_Security_Toolkit, and reused in Stage 2 to derive a
soil-moisture proxy feature for the flood-risk classifier.

Structure: 1 soil-moisture store + 2 linear reservoirs (quickflow,
baseflow). This is deliberately a simple linear-reservoir cascade, not
GR4J -- calibrated per catchment via differential evolution maximizing
NSE on the training period. Calibrated parameters are saved as JSON
checkpoints so downstream scripts (evaluate.py, Stage 2's
build_features.py) don't need to recalibrate.

Usage:
    python src/stage1_lstm_forecast/conceptual_baseline.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from data_utils import load_catchment_split, CATCHMENT_AREA_KM2

WARMUP_DAYS = 365  # excluded from NSE so store spin-up doesn't bias the fit

PARAM_BOUNDS = [
    (50.0, 500.0),   # s_max_mm: soil moisture store capacity
    (0.0, 1.0),      # alpha: fraction of excess routed to quickflow (rest to baseflow)
    (1.0, 10.0),     # k_quick_days: quickflow reservoir residence time
    (10.0, 200.0),   # k_base_days: baseflow reservoir residence time
]
PARAM_NAMES = ["s_max_mm", "alpha", "k_quick_days", "k_base_days"]

MODEL_DIR = Path(
    "/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit"
    "/models/stage1_conceptual_baseline"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def simulate_discharge(
    precip_mm: np.ndarray,
    pet_mm: np.ndarray,
    params,
    return_states: bool = False,
):
    """
    Run the 3-store linear-reservoir model forward.

    If return_states is False (default, used by Stage 1): returns q_sim_mm only.
    If return_states is True (used by Stage 2 for the soil-moisture feature):
    returns (q_sim_mm, soil_store_mm) -- the soil-moisture store trajectory
    alongside discharge.
    """
    s_max_mm, alpha, k_quick_days, k_base_days = params
    n = len(precip_mm)

    soil_store = s_max_mm * 0.5  # initial condition: half full
    quick_store = 0.0
    base_store = 0.0

    q_sim_mm = np.zeros(n)
    soil_store_mm = np.zeros(n) if return_states else None

    for t in range(n):
        actual_et = pet_mm[t] * (soil_store / s_max_mm) if s_max_mm > 0 else 0.0
        soil_temp = soil_store + precip_mm[t] - actual_et

        if soil_temp > s_max_mm:
            excess = soil_temp - s_max_mm
            soil_store = s_max_mm
        else:
            excess = 0.0
            soil_store = max(soil_temp, 0.0)

        quick_input = alpha * excess
        base_input = (1.0 - alpha) * excess

        quick_store = quick_store * (1.0 - 1.0 / k_quick_days) + quick_input / k_quick_days
        base_store = base_store * (1.0 - 1.0 / k_base_days) + base_input / k_base_days

        q_sim_mm[t] = quick_store + base_store
        if return_states:
            soil_store_mm[t] = soil_store

    if return_states:
        return q_sim_mm, soil_store_mm
    return q_sim_mm


def nash_sutcliffe_efficiency(q_obs_mm: np.ndarray, q_sim_mm: np.ndarray) -> float:
    """
    Standard NSE: 1 - sum((obs-sim)^2) / sum((obs-mean(obs))^2).
    Days with missing observed discharge are excluded from the calculation.
    """
    valid = ~np.isnan(q_obs_mm)
    q_obs_valid = q_obs_mm[valid]
    q_sim_valid = q_sim_mm[valid]
    numerator = np.sum((q_obs_valid - q_sim_valid) ** 2)
    denominator = np.sum((q_obs_valid - np.mean(q_obs_valid)) ** 2)
    return 1.0 - numerator / denominator


def kling_gupta_efficiency(q_obs_mm: np.ndarray, q_sim_mm: np.ndarray) -> float:
    """
    KGE (2009 formulation): correlation, variability ratio, bias ratio combined.
    Days with missing observed discharge are excluded from the calculation.
    """
    valid = ~np.isnan(q_obs_mm)
    q_obs_valid = q_obs_mm[valid]
    q_sim_valid = q_sim_mm[valid]
    r = np.corrcoef(q_obs_valid, q_sim_valid)[0, 1]
    alpha_ratio = np.std(q_sim_valid) / np.std(q_obs_valid)
    beta_ratio = np.mean(q_sim_valid) / np.mean(q_obs_valid)
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha_ratio - 1) ** 2 + (beta_ratio - 1) ** 2)


def _objective(params, precip_mm, pet_mm, q_obs_mm) -> float:
    """Objective for differential_evolution: negative NSE, warm-up excluded."""
    q_sim_mm = simulate_discharge(precip_mm, pet_mm, params)
    nse = nash_sutcliffe_efficiency(q_obs_mm[WARMUP_DAYS:], q_sim_mm[WARMUP_DAYS:])
    return -nse


def calibrate(train_df: pd.DataFrame, maxiter: int = 200, seed: int = 42) -> dict:
    """
    Calibrate the 4 model parameters on a catchment's training period via
    differential evolution, maximizing NSE.
    """
    precip_mm = train_df["precipitation"].to_numpy()
    pet_mm = train_df["pet"].to_numpy()
    q_obs_mm = train_df["discharge_spec"].to_numpy()

    result = differential_evolution(
        _objective,
        bounds=PARAM_BOUNDS,
        args=(precip_mm, pet_mm, q_obs_mm),
        seed=seed,
        maxiter=maxiter,
        polish=True,
        workers=-1,
        disp=True,
    )

    params = dict(zip(PARAM_NAMES, result.x))
    params["train_nse"] = -result.fun
    params["iterations_used"] = result.nit
    return params


def save_checkpoint(station_id: str, params: dict) -> None:
    """
    Save calibrated baseline parameters as JSON. Unlike the LSTM, there
    are no model weights here -- just 4 scalar parameters -- so JSON is
    simpler and more inspectable than a torch/pickle checkpoint.
    """
    checkpoint_path = MODEL_DIR / f"baseline_{station_id}.json"
    serializable_params = {k: float(v) for k, v in params.items()}
    with open(checkpoint_path, "w") as f:
        json.dump(serializable_params, f, indent=2)
    print(f"  Saved checkpoint: {checkpoint_path}")


def load_checkpoint(station_id: str) -> dict:
    """Load previously calibrated baseline parameters for a station."""
    checkpoint_path = MODEL_DIR / f"baseline_{station_id}.json"
    with open(checkpoint_path) as f:
        return json.load(f)


def evaluate(df: pd.DataFrame, params: dict) -> dict:
    """Run the calibrated model on a period (val/test) and return NSE and KGE."""
    precip_mm = df["precipitation"].to_numpy()
    pet_mm = df["pet"].to_numpy()
    q_obs_mm = df["discharge_spec"].to_numpy()

    param_values = [params[name] for name in PARAM_NAMES]
    q_sim_mm = simulate_discharge(precip_mm, pet_mm, param_values)

    nse = nash_sutcliffe_efficiency(q_obs_mm[WARMUP_DAYS:], q_sim_mm[WARMUP_DAYS:])
    kge = kling_gupta_efficiency(q_obs_mm[WARMUP_DAYS:], q_sim_mm[WARMUP_DAYS:])
    return {"nse": nse, "kge": kge}


if __name__ == "__main__":
    FULL_CALIBRATION_MAXITER = 150

    for station_id in CATCHMENT_AREA_KM2:
        print(f"Calibrating conceptual baseline for station {station_id} "
              f"(maxiter={FULL_CALIBRATION_MAXITER})...")
        split = load_catchment_split(station_id)

        params = calibrate(split.train, maxiter=FULL_CALIBRATION_MAXITER)
        save_checkpoint(station_id, params)
        print(f"  Calibrated params: { {k: round(v, 3) for k, v in params.items()} }")

        val_scores = evaluate(split.val, params)
        test_scores = evaluate(split.test, params)
        print(f"  Validation: NSE={val_scores['nse']:.3f}, KGE={val_scores['kge']:.3f}")
        print(f"  Test:       NSE={test_scores['nse']:.3f}, KGE={test_scores['kge']:.3f}")
        print()