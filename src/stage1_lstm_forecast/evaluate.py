"""
evaluate.py

Loads the calibrated conceptual baseline (from its saved JSON checkpoint)
and saved LSTM checkpoints for both catchments, and produces the Stage 1
comparison outputs: observed vs. simulated hydrographs over the test
period, and a summary NSE/KGE table -- baseline vs. LSTM, both catchments.

Figures are saved to results/stage1_lstm_forecast/, not just displayed,
so they're reusable portfolio assets. Neither model is retrained or
recalibrated here -- both load from checkpoints saved by
conceptual_baseline.py and lstm_model.py respectively.

Usage:
    python src/stage1_lstm_forecast/evaluate.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import load_catchment_split, load_catchment_timeseries, mmday_to_cumecs, CATCHMENT_AREA_KM2, VAL_END
from conceptual_baseline import simulate_discharge, PARAM_NAMES, load_checkpoint
from lstm_model import RainfallRunoffLSTM, build_windows, normalize, denormalize_target, MODEL_DIR, DEVICE

RESULTS_DIR = Path(
    "/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit"
    "/results/stage1_lstm_forecast"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATION_LABELS = {
    "39002": "Thames at Days Weir",
    "76007": "Eden at Sheepmount",
}


def load_lstm_checkpoint(station_id: str) -> tuple[RainfallRunoffLSTM, dict]:
    """Load a saved LSTM checkpoint and reconstruct the model for inference."""
    checkpoint_path = MODEL_DIR / f"lstm_{station_id}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model = RainfallRunoffLSTM(
        input_size=len(checkpoint["feature_cols"]),
        hidden_size=checkpoint["hidden_size"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def get_lstm_test_predictions(station_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct test-period predictions from a saved LSTM checkpoint.
    Returns (dates, observed_mm, predicted_mm).
    """
    model, checkpoint = load_lstm_checkpoint(station_id)
    stats = checkpoint["stats"]

    full_df = load_catchment_timeseries(station_id)
    test_dates = full_df.loc[VAL_END:].index[1:]

    norm_df = normalize(full_df, stats)
    norm_df["discharge_spec_norm"] = (full_df["discharge_spec"] - stats["target_mean"]) / stats["target_std"]
    build_df = norm_df.assign(discharge_spec=norm_df["discharge_spec_norm"])

    x_test, y_test_norm = build_windows(build_df, test_dates)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_norm = model(x_test_t).cpu().numpy().flatten()

    pred_mm = denormalize_target(pred_norm, stats)
    obs_mm = denormalize_target(y_test_norm, stats)

    # dates aligned to y_test (windows with NaN targets already dropped)
    valid_dates = full_df.loc[VAL_END:].index[1:][~full_df.loc[VAL_END:]["discharge_spec"][1:].isna().to_numpy()]
    valid_dates = valid_dates[-len(pred_mm):]  # align to dropped-lookback windows

    return valid_dates, obs_mm, pred_mm


def get_baseline_test_predictions(station_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the saved conceptual baseline checkpoint and return test-period predictions."""
    split = load_catchment_split(station_id)
    params = load_checkpoint(station_id)

    param_values = [params[name] for name in PARAM_NAMES]
    precip_mm = split.test["precipitation"].to_numpy()
    pet_mm = split.test["pet"].to_numpy()
    sim_mm = simulate_discharge(precip_mm, pet_mm, param_values)

    obs_mm = split.test["discharge_spec"].to_numpy()
    dates = split.test.index.to_numpy()

    return dates, obs_mm, sim_mm


def plot_hydrograph(station_id: str) -> None:
    """Overlay observed, baseline, and LSTM discharge for a catchment's test period."""
    label = STATION_LABELS[station_id]
    area_km2 = CATCHMENT_AREA_KM2[station_id]

    base_dates, base_obs_mm, base_sim_mm = get_baseline_test_predictions(station_id)
    lstm_dates, lstm_obs_mm, lstm_sim_mm = get_lstm_test_predictions(station_id)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(base_dates, mmday_to_cumecs(base_obs_mm, area_km2), label="Observed", color="black", linewidth=1)
    ax.plot(base_dates, mmday_to_cumecs(base_sim_mm, area_km2), label="Conceptual baseline", color="tab:blue", alpha=0.7)
    ax.plot(lstm_dates, mmday_to_cumecs(lstm_sim_mm, area_km2), label="LSTM", color="tab:orange", alpha=0.7)

    ax.set_title(f"{label} ({station_id}) -- test period discharge")
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (m3/s)")
    ax.legend()
    fig.tight_layout()

    output_path = RESULTS_DIR / f"hydrograph_{station_id}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    for station_id in CATCHMENT_AREA_KM2:
        print(f"Generating hydrograph for station {station_id}...")
        plot_hydrograph(station_id)