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
from sklearn.preprocessing import StandardScaler

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline,
    AnomalyScoreMethods,
)

from ._hf_cache import ensure_offline_env, resolve_local_model_path, model_load_lock
from .base import BaseDetector

# Set offline mode BEFORE any from_pretrained call so the loader never pings
# huggingface.co (avoids multi-second SSL timeouts + meta-tensor corruption).
ensure_offline_env()

# Model constants — sourced from the central registry (single source of
# truth).  The names are kept for backwards-compatible imports
# (``from phm.algorithm.tspulse import DEFAULT_MODEL`` still works).
from ._registry import get_model_entry as _get_entry

_TSP_ENTRY = _get_entry("tspulse")
DEFAULT_MODEL = _TSP_ENTRY.hub_id
CONTEXT_LENGTH = _TSP_ENTRY.context_length


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

        # Resolve hub id → local snapshot path so the loader stays offline.
        raw_path = model_path or DEFAULT_MODEL
        path = resolve_local_model_path(raw_path) or raw_path
        load_kwargs = {}
        # If path is a local directory, don't pass revision
        if path and os.path.isdir(path):
            load_kwargs = {}
        else:
            load_kwargs = {"revision": model_revision}

        # Serialise model construction (see _hf_cache.model_load_lock docstring).
        with model_load_lock:
            self.model = TSPulseForReconstruction.from_pretrained(path, **load_kwargs)
        self.model = self.model.to(device).float().eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.model_source = path

    def detect(self, values, train_values_for_scaler=None, context=None):
        """Run anomaly detection on a 1-D telemetry array.

        Args:
            values: np.ndarray [T] float32 — telemetry values to score
            train_values_for_scaler: np.ndarray or None — training data for StandardScaler
            context: optional preceding block (np.ndarray [C]) prepended to
                ``values`` before pipeline inference to give the pipeline's
                aggregation/smoothing enough context.  Only the last T scores
                (corresponding to ``values``) are returned.  Without context,
                the pipeline produces near-zero scores on short blocks of
                slowly-varying channels.

        Returns:
            scores: np.ndarray [T] float32 — anomaly scores **clipped to
            ``[0, 1]``** (higher = more anomalous).  The clip preserves the
            absolute reconstruction-error magnitude and aligns the score with
            the global ``ANOMALY_THRESHOLD = 0.5``; it also keeps the
            downstream direction-flip (``1 - score``) well-defined.  On
            constant-score inputs (e.g. all-zero) the raw values are returned
            unchanged.
        """
        # Standardize
        n_target = len(values)
        if train_values_for_scaler is not None:
            scaler = StandardScaler().fit(train_values_for_scaler.reshape(-1, 1))
        else:
            scaler = StandardScaler().fit(values.reshape(-1, 1))
        scaled = scaler.transform(values.reshape(-1, 1)).flatten().astype(np.float32)

        # Prepend context (also standardized) for pipeline inference.
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

        # Run inference under no_grad — eval mode alone disables training,
        # but wrapping the forward pass avoids building any autograd graph,
        # which is both faster (Phase 0 measured serial 1.69s→0.44s, 3.8x)
        # and safer under concurrent ThreadPoolExecutor access.
        with torch.no_grad():
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

        # Trim to target block: drop context-prefix scores and padding.
        if context_len > 0:
            scores = scores[context_len:]
        if len(scores) > n_target:
            scores = scores[-n_target:]

        # Clip to [0, 1].  Per-window MinMax was removed: it forced every
        # window's max to 1.0, causing false alarms on normal periodic
        # waveforms whose relative-max reconstruction error is small but got
        # stretched to 1.0.  Pipeline output (standardised MSE) is already in
        # meaningful units — normal stays low (~0.3-0.4), anomalies stand out.
        return np.clip(scores, 0.0, 1.0).astype(np.float32)


__all__ = ["AnomalyDetector", "DEFAULT_MODEL", "CONTEXT_LENGTH"]
