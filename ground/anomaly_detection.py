"""Anomaly detection module using TSPulse (zero-shot reconstruction-based).

This module wraps the TSPulse pre-trained model to provide real-time anomaly
scoring on telemetry channels. It is designed to simulate the "space segment"
(lightweight on-orbit inference) in the space-ground collaborative architecture.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline,
    AnomalyScoreMethods,
)

# Model constants
DEFAULT_MODEL = "ibm-granite/granite-timeseries-tspulse-r1"
CONTEXT_LENGTH = 512


class AnomalyDetector:
    """TSPulse-based anomaly detector for single-channel telemetry.

    Args:
        device: "cuda" or "cpu"
        model_path: HuggingFace model name or local directory path.
                   If None, uses the default online model.
                   For fine-tuned models, pass the local checkpoint directory.
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
            scores: np.ndarray [T] float32 — anomaly scores (higher = more anomalous)
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
            smoothing_length=1,
        )

        result = pipeline(df)
        raw_scores = result["anomaly_score"].iloc[0]
        if isinstance(raw_scores, (list, np.ndarray)):
            raw_scores = np.array(raw_scores, dtype=np.float32).flatten()
        else:
            raw_scores = result["anomaly_score"].values.astype(np.float32)

        # Align to original input length
        n_out = min(len(raw_scores), T)
        scores = np.zeros(T, dtype=np.float32)
        scores[:n_out] = raw_scores[:n_out]

        # Trim padding (both front-padding from tiling and back-padding from +1)
        if len(scores) > len(values):
            scores = scores[-len(values):]
        return scores
