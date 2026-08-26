"""
lstm_model.py

LSTM rainfall-runoff forecaster for Stage 1 of the Flood_and_Water_Security_Toolkit.
Trained independently per catchment, benchmarked against the conceptual
baseline in conceptual_baseline.py using the same NSE/KGE functions so
the comparison is on identical footing. Saves a checkpoint per catchment
(weights + normalization stats + architecture config) for reuse in
evaluate.py without retraining.

Usage:
    python src/stage1_lstm_forecast/lstm_model.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from data_utils import load_catchment_timeseries, CATCHMENT_AREA_KM2, TRAIN_END, VAL_END
from conceptual_baseline import nash_sutcliffe_efficiency, kling_gupta_efficiency

FEATURE_COLS = ["precipitation", "pet", "temperature"]
TARGET_COL = "discharge_spec"
LOOKBACK_DAYS = 365
WARMUP_DAYS = 365  # matches conceptual_baseline.py, for a fair comparison

HIDDEN_SIZE = 64
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5  # early stopping: epochs without val NSE improvement
LEARNING_RATE = 1e-3
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_DIR = Path(
    "/Users/ShreyaJariwalaMain/_GeoAI_Notebook/Flood_and_Water_Security_Toolkit"
    "/models/stage1_lstm"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """
    Fix all sources of randomness for reproducible training runs: Python's
    random module, numpy, torch's CPU RNG, and torch's MPS RNG if available.
    Without this, weight initialization and DataLoader shuffling differ
    every run, making test NSE swings impossible to distinguish from real
    training improvements.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class SequenceDataset(Dataset):
    """Wraps precomputed (X, y) window arrays as a PyTorch Dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RainfallRunoffLSTM(nn.Module):
    """Single-layer LSTM -> dense output, predicting next-day discharge."""

    def __init__(self, input_size: int = 3, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.output_layer(h_n[-1])


def build_windows(full_df: pd.DataFrame, target_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sequence windows for a set of target dates. Each window
    uses the LOOKBACK_DAYS days strictly before the target date as input
    features. Windows whose target discharge is NaN, or whose lookback
    would extend before the start of the record, are dropped.
    """
    feature_array = full_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    target_array = full_df[TARGET_COL].to_numpy(dtype=np.float32)
    date_to_pos = {date: pos for pos, date in enumerate(full_df.index)}

    x_windows, y_targets = [], []
    for date in target_dates:
        pos = date_to_pos[date]
        if pos - LOOKBACK_DAYS < 0:
            continue
        target = target_array[pos]
        if np.isnan(target):
            continue
        window = feature_array[pos - LOOKBACK_DAYS: pos]
        x_windows.append(window)
        y_targets.append(target)

    return np.stack(x_windows), np.array(y_targets, dtype=np.float32)


def compute_train_stats(train_df: pd.DataFrame) -> dict:
    """Z-score normalization statistics, computed from the training period only."""
    stats = {}
    for col in FEATURE_COLS:
        stats[f"{col}_mean"] = train_df[col].mean()
        stats[f"{col}_std"] = train_df[col].std()
    stats["target_mean"] = train_df[TARGET_COL].mean()
    stats["target_std"] = train_df[TARGET_COL].std()
    return stats


def normalize(full_df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply train-period z-score stats to the full series (all splits)."""
    df = full_df.copy()
    for col in FEATURE_COLS:
        df[col] = (df[col] - stats[f"{col}_mean"]) / stats[f"{col}_std"]
    return df


def denormalize_target(y_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Convert normalized discharge predictions back to mm/day."""
    return y_norm * stats["target_std"] + stats["target_mean"]


def train_lstm(station_id: str) -> dict:
    """
    Train and evaluate an LSTM for a single catchment, returning val/test
    NSE and KGE alongside the fitted model and normalization stats. Saves
    a checkpoint (weights + stats + architecture config) to MODEL_DIR.
    """
    full_df = load_catchment_timeseries(station_id)
    train_dates = full_df.loc[:TRAIN_END].index
    val_dates = full_df.loc[TRAIN_END:VAL_END].index[1:]
    test_dates = full_df.loc[VAL_END:].index[1:]

    stats = compute_train_stats(full_df.loc[:TRAIN_END])
    norm_df = normalize(full_df, stats)
    # Target is normalized too for training loss, but we keep the raw
    # (non-normalized) target column separately for NSE/KGE evaluation.
    norm_df[f"{TARGET_COL}_norm"] = (full_df[TARGET_COL] - stats["target_mean"]) / stats["target_std"]

    def windows_for(dates):
        raw_targets = build_windows(norm_df.assign(**{TARGET_COL: norm_df[f"{TARGET_COL}_norm"]}), dates)
        return raw_targets

    x_train, y_train = windows_for(train_dates[WARMUP_DAYS:])
    x_val, y_val = windows_for(val_dates)
    x_test, y_test = windows_for(test_dates)

    print(f"  Windows -- train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")

    # generator pinned to SEED so DataLoader shuffling is reproducible too
    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )

    model = RainfallRunoffLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(DEVICE)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

    best_val_nse = -np.inf
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_norm = model(x_val_t).cpu().numpy().flatten()
        val_pred_mm = denormalize_target(val_pred_norm, stats)
        val_obs_mm = denormalize_target(y_val, stats)
        val_nse = nash_sutcliffe_efficiency(val_obs_mm, val_pred_mm)

        print(f"  Epoch {epoch}: val NSE = {val_nse:.4f}")

        if val_nse > best_val_nse:
            best_val_nse = val_nse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val NSE = {best_val_nse:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_norm = model(x_test_t).cpu().numpy().flatten()
    test_pred_mm = denormalize_target(test_pred_norm, stats)
    test_obs_mm = denormalize_target(y_test, stats)
    test_nse = nash_sutcliffe_efficiency(test_obs_mm, test_pred_mm)
    test_kge = kling_gupta_efficiency(test_obs_mm, test_pred_mm)

    checkpoint_path = MODEL_DIR / f"lstm_{station_id}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "hidden_size": HIDDEN_SIZE,
            "lookback_days": LOOKBACK_DAYS,
            "feature_cols": FEATURE_COLS,
            "best_val_nse": best_val_nse,
            "test_nse": test_nse,
            "test_kge": test_kge,
            "seed": SEED,
        },
        checkpoint_path,
    )
    print(f"  Saved checkpoint: {checkpoint_path}")

    return {
        "station_id": station_id,
        "best_val_nse": best_val_nse,
        "test_nse": test_nse,
        "test_kge": test_kge,
        "model": model,
        "stats": stats,
    }


if __name__ == "__main__":
    set_seed()
    print(f"Using device: {DEVICE}\n")

    for station_id in CATCHMENT_AREA_KM2:
        print(f"Training LSTM for station {station_id}...")
        results = train_lstm(station_id)
        print(f"  Best val NSE: {results['best_val_nse']:.3f}")
        print(f"  Test NSE:     {results['test_nse']:.3f}")
        print(f"  Test KGE:     {results['test_kge']:.3f}")
        print()