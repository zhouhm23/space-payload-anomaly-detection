"""Anomaly detection plugin — TSPulse (zero-shot reconstruction-based).

Migrated verbatim from the legacy ``ground/anomaly_detection.py``.  The
class name (``AnomalyDetector``) and method signature are unchanged so
existing import sites keep working.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline,
    AnomalyScoreMethods,
)

from .base import BaseDetector

# Model constants — preserved for backwards-compatible ``from ... import
# CONTEXT_LENGTH`` callers (e.g. ground/tests/test_models.py).
DEFAULT_MODEL = "ibm-granite/granite-timeseries-tspulse-r1"
CONTEXT_LENGTH = 512


class AnomalyDetector(BaseDetector):
    """TSPulse-based anomaly detector for single-channel telemetry.

    Args:
        device: "cuda" or "cpu"
        model_path: HuggingFace model name or local directory path.
                   If None, uses the default online model.
        model_revision: HuggingFace revision (ignored if model_path is local)

    Usage:
        detector = AnomalyDetector(device="cuda")
        scores = detector.detect(values, train_values_for_scaler)
    """

    def __init__(self, device="cuda", model_path=None, model_revision="main"):
        self.device = device
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        path = model_path or DEFAULT_MODEL
        load_kwargs = {}
        # If path is a local directory, don't pass revision
        if path and os.path.isdir(path):
            load_kwargs = {}
        else:
            load_kwargs = {"revision": model_revision}

        self.model = TSPulseForReconstruction.from_pretrained(path, **load_kwargs)
        self.model = self.model.to(device).float().eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.model_source = path

    def detect(self, values, train_values_for_scaler=None):
        """Run anomaly detection on a 1-D telemetry array.

        Args:
            values: np.ndarray [T] float32 — telemetry values to score
            train_values_for_scaler: np.ndarray or None — training data for StandardScaler

        Returns:
            scores: np.ndarray [T] float32 — anomaly scores **MinMax-normalised
            to [0, 1]** (higher = more anomalous).  The normalisation aligns
            the score with the global ``ANOMALY_THRESHOLD = 0.7`` and makes
            the downstream direction-flip (``1 - score``) well-defined.  On
            constant-score inputs (e.g. all-zero) the raw values are returned
            unchanged.
        """
        # Standardize
        if train_values_for_scaler is not None:
            scaler = StandardScaler().fit(train_values_for_scaler.reshape(-1, 1))
        else:
            scaler = StandardScaler().fit(values.reshape(-1, 1))
        scaled = scaler.transform(values.reshape(-1, 1)).flatten().astype(np.float32)

        T = len(scaled)
        # Ensure enough points: tile if shorter than one window
        if T < CONTEXT_LENGTH:
            repeats = (CONTEXT_LENGTH // T) + 1
            scaled = np.tile(scaled, repeats)
            T = len(scaled)

        # Pipeline aggregation has an off-by-one on exact window boundaries
        # (produces N*512 + 1 scores for N*512 inputs). Avoid by adding one
        # extra point, then trimming output.
        if T % CONTEXT_LENGTH == 0:
            scaled = np.concatenate([scaled, scaled[-1:]])
            T = len(scaled)

        n_windows = T // CONTEXT_LENGTH

        # Build pipeline once
        df = pd.DataFrame({"x": scaled})
        df["timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="s")

        pipeline = TimeSeriesAnomalyDetectionPipeline(
            self.model,
            timestamp_column="timestamp",
            target_columns=["x"],
            prediction_mode=[
                AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
                AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            ],
            aggregation_length=64,
            aggr_function="max",
            smoothing_length=8,
            least_significant_scale=0.01,
            least_significant_score=0.1,
        )

        result = pipeline(df)
        # Pipeline returns anomaly_score as a column whose first row holds the
        # per-sample array.  Read it defensively: .iloc[0] may return a scalar
        # on degenerate single-window outputs, so flatten whatever shape we
        # get and let the length-alignment below trim/pad as needed.
        col = result["anomaly_score"]
        if hasattr(col, "iloc"):
            first = col.iloc[0]
        else:
            first = col[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            raw_scores = np.array(first, dtype=np.float32).ravel()
        else:
            # Scalar (single-window degenerate case) — fall back to the
            # full column so we keep one value per output sample.
            raw_scores = np.asarray(col, dtype=np.float32).ravel()

        # Align to original input length
        n_out = min(len(raw_scores), T)
        scores = np.zeros(T, dtype=np.float32)
        scores[:n_out] = raw_scores[:n_out]

        # Trim padding (both front-padding from tiling and back-padding from +1)
        if len(scores) > len(values):
            scores = scores[-len(values):]

        # MinMax-normalise to [0, 1] so the score is comparable to the
        # ANOMALY_THRESHOLD (0.7) configured in phm.config, and so the
        # direction-flip (1 - score) downstream is well-defined.  Matches
        # the eval-pipeline convention (experiments/tspulse_eval/*).
        rng = float(scores.max() - scores.min())
        if rng > 1e-12:  # avoid division by zero on constant-score inputs
            scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
        return scores.astype(np.float32)


__all__ = ["AnomalyDetector", "DEFAULT_MODEL", "CONTEXT_LENGTH"]
