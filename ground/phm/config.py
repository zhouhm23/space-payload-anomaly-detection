"""Centralised PHM configuration constants.

Keeping thresholds in one place avoids magic numbers scattered across the
alarm / warning / health services.
"""

from __future__ import annotations

# ── Thresholds ──────────────────────────────────────────────────────────────
# Anomaly score above which a sample is considered anomalous.  Aligns with
# the space-side alert trigger (space/main.py ALERT_THRESHOLD).  Tuned at
# 0.5 for clip-normalised pipeline scores (see config notes): normal
# periodic waveforms stay below 0.5 while genuine anomalies exceed it.
ANOMALY_THRESHOLD: float = 0.5

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

# ── Cascade layer configuration ─────────────────────────────────────────────
# Layer 1 classic filter
L1_CONSTANT_STD: float = 1e-3       # channels with std < this are "constant"
L1_SIGMA_K: float = 3.0             # 3σ rule multiplier
L1_IQR_FACTOR: float = 1.5          # IQR fence multiplier
# Layer 3 physical constraint (statistical defaults — override with real
# payload domain knowledge when available)
L3_CONSTANT_STD: float = 1e-3       # suppress scores on constant channels
L3_RANGE_BOOST: float = 0.95        # boost score to this when out-of-range
L3_RATE_BOOST: float = 0.85         # boost score when rate-of-change exceeds limit

# ── SQLite persistence ──────────────────────────────────────────────────────
SQLITE_ENABLED: bool = True
SQLITE_BATCH_SIZE: int = 200        # flush after this many queued items
SQLITE_FLUSH_INTERVAL: float = 2.0  # or after this many seconds
