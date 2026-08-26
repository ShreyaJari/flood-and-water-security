"""
check_alignment.py

One-off diagnostic: confirms get_lstm_test_predictions() returns
consistently-sized, correctly-dated arrays before trusting the
Stage 1 hydrograph plots.
"""

from evaluate import get_lstm_test_predictions

for station_id in ["39002", "76007"]:
    dates, obs, pred = get_lstm_test_predictions(station_id)
    print(station_id, "dates:", len(dates), "obs:", len(obs), "pred:", len(pred))
    print("  first date:", dates[0], " last date:", dates[-1])