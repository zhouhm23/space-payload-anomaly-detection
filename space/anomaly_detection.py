"""Anomaly detection module using TSPulse (zero-shot reconstruction-based).

This module wraps the TSPulse pre-trained model to provide real-time anomaly
scoring on telemetry channels. It is designed to simulate the "space segment"
(lightweight on-orbit inference) in the space-ground collaborative architecture.

Algorithm is kept **identical** to the ground segment
(``ground/phm/algorithm/tspulse.py``) so that both segments produce
comparable scores on the same input.  The only difference is that this
module does not inherit from ``BaseDetector`` (the space segment is a
standalone process and does not import the ground PHM package).
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

    def detect(self, values, train_values_for_scaler=None, context=None):
        """Run anomaly detection on a 1-D telemetry array.

        Args:
            values: np.ndarray [T] float32 — telemetry values to score.
            train_values_for_scaler: optional training data for StandardScaler.
            context: optional preceding block (np.ndarray [C]) prepended to
                ``values`` before pipeline inference to give the pipeline's
                aggregation/smoothing enough context.  Only the last T scores
                (corresponding to ``values``) are returned.  Without context,
                the pipeline's internal aggregation produces near-zero scores
                on short (512-point) blocks of slowly-varying channels —
                overlap fixes this.

        Returns:
            scores: np.ndarray [T] float32 — anomaly scores clipped to [0, 1].
            Pipeline output (standardised MSE) is preserved as-is so normal
            periodic waveforms keep a low score (~0.3-0.4) while genuine
            anomalies stand out (>0.5).
        """
        n_target = len(values)
        # Standardize (fit on values or train, NOT context — context is only
        # for pipeline context, scale must match the target block).
        if train_values_for_scaler is not None:
            scaler = StandardScaler().fit(train_values_for_scaler.reshape(-1, 1))
        else:
            scaler = StandardScaler().fit(values.reshape(-1, 1))
        scaled = scaler.transform(values.reshape(-1, 1)).flatten().astype(np.float32)

        # Prepend context (also standardized) for pipeline inference, then
        # trim the output to keep only the target block's scores.
        context_len = 0
        if context is not None and len(context) > 0:
            ctx_scaled = scaler.transform(np.asarray(context, dtype=np.float32).reshape(-1, 1)).flatten().astype(np.float32)
            scaled = np.concatenate([ctx_scaled, scaled])
            context_len = len(ctx_scaled)

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

        # Build pipeline (same parameters as ground segment)
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
        # per-sample array.  Read defensively: .iloc[0] may return a scalar
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

        # Trim to target block: drop the context-prefix scores and any
        # padding (front-padding from tiling, back-padding from +1).
        if context_len > 0:
            scores = scores[context_len:]
        if len(scores) > n_target:
            scores = scores[-n_target:]

        # Clip to [0, 1].  Pipeline output is already in roughly standardised
        # units (StandardScaler'd input → MSE), so clip preserves the absolute
        # error magnitude.  Per-window MinMax normalisation was removed because
        # it forced every window's max to 1.0, causing false alarms on normal
        # periodic waveforms (sine/multi_sine) whose relative-max error is
        # small but gets stretched to 1.0.  With clip, a normal sine block
        # stays ~0.3-0.4 while genuine anomalies (T-4 ~0.68) stand out.
        return np.clip(scores, 0.0, 1.0).astype(np.float32)


__all__ = ["AnomalyDetector", "DEFAULT_MODEL", "CONTEXT_LENGTH"]
