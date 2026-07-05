"""Centralised PHM configuration constants.

Keeping thresholds in one place avoids magic numbers scattered across the
alarm / warning / health services.
"""

from __future__ import annotations

# ── Thresholds ──────────────────────────────────────────────────────────────
# Anomaly score above which a sample is considered anomalous.  Aligns with
# the 0.7 reference line drawn on the anomaly chart and with the space-side
# alert trigger (space/comm.py).
ANOMALY_THRESHOLD: float = 0.7

# Forecaster (TTM-R3) context / prediction lengths.  Mirror the values used
# by ``algorithm/ttm.py`` so callers do not need to import the model module
# just to read these.
FORECAST_CONTEXT_LENGTH: int = 512
FORECAST_PREDICTION_LENGTH: int = 96

# ── Ring buffer sizing ──────────────────────────────────────────────────────
RING_BUFFER_MAX: int = 20000

# ── Warning lifecycle ───────────────────────────────────────────────────────
# How many recent ground-detected prediction-segment scores define the
# "predicted anomaly" — used by the warning service to decide whether to
# emit a new early-warning entry.
WARNING_MIN_PREDICT_SCORES: int = 1
