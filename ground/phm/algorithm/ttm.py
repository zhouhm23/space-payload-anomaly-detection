"""Forecasting plugin — TTM-R3 (zero-shot prediction).

Migrated verbatim from the legacy ``ground/forecasting.py``.  The class
name (``TrendForecaster``) and method signature are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

from ._hf_cache import ensure_offline_env, model_load_lock, resolve_local_model_path
from .base import BaseForecaster

# Set offline mode BEFORE any from_pretrained call so the loader never pings
# huggingface.co (avoids multi-second SSL timeouts + meta-tensor corruption).
ensure_offline_env()

# Model constants — preserved for backwards-compatible imports.
DEFAULT_MODEL = "ibm-research/ttm-r3"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96


class TrendForecaster(BaseForecaster):
    """TTM-R3-based forecaster for single-channel telemetry.

    Args:
        device: "cuda" or "cpu"
        model_path: HuggingFace model name or local directory path.
                   If None, uses the default online model.
                   For fine-tuned models, pass the local checkpoint directory.

    Usage:
        forecaster = TrendForecaster(device="cuda")
        pred, ctx = forecaster.forecast(values, train_values_for_scaler)
    """

    def __init__(self, device="cuda", model_path=None):
        self.device = device

        path = model_path or DEFAULT_MODEL
        # Serialise model construction: get_model / from_pretrained use
        # non-thread-safe module-level init hooks; concurrent calls corrupt
        # torch's parameter state and produce meta tensors.
        with model_load_lock:
            self.model = self._load_model(path)
        self.model = self.model.to(device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.model_source = path

    @staticmethod
    def _load_model(path: str):
        """Load TTM-R3 with the correct architecture class, fully offline.

        ``tsfm_public.toolkit.get_model.get_model`` knows how to pick the right
        model class + revision for a given (context_length, prediction_length),
        but it ignores ``HF_HUB_OFFLINE`` and always pings huggingface.co —
        which fails with an SSL error on this machine and corrupts model
        construction (meta tensors).  We replicate its selection logic for the
        one configuration this system uses (context=512, prediction=96 on
        ``ibm-research/ttm-r3``) and call ``from_pretrained`` directly with the
        resolved revision, so the HF cache is read without any network access.

        The selection rule (from get_model source): for r3 models the chosen
        revision contains "-dec-" → use ``TinyTimeMixerForDecomposedPrediction``;
        otherwise ``TinyTimeMixerForPrediction``.  Using the wrong class leaves
        weights randomly initialised and produces jagged garbage predictions.
        """
        # The r3 revision for context=512, prediction=96 is "512-96-dec-512-r3".
        # It contains "-dec-" → decomposed prediction variant.
        # Resolve to a local snapshot path so from_pretrained never pings the
        # hub (a hub id + revision forces a network round-trip even with
        # HF_HUB_OFFLINE set, because transformers must confirm the revision).
        local_path = resolve_local_model_path(path) or path
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerForDecomposedPrediction,
        )
        return TinyTimeMixerForDecomposedPrediction.from_pretrained(
            local_path,
            prediction_filter_length=PREDICTION_LENGTH,
        )

    def forecast(self, values, train_values_for_scaler=None):
        """Forecast future PREDICTION_LENGTH steps from the last CONTEXT_LENGTH steps.

        Args:
            values: np.ndarray [T] float32 — telemetry values (T >= CONTEXT_LENGTH)
            train_values_for_scaler: np.ndarray or None — for StandardScaler fitting

        Returns:
            context_raw: np.ndarray [CONTEXT_LENGTH] — the input window in **original** scale
            prediction_raw: np.ndarray [PREDICTION_LENGTH] — forecasted values in **original** scale
            scaler: the fitted StandardScaler (for additional inverse_transform if needed)
        """
        # Standardize
        if train_values_for_scaler is not None:
            scaler = StandardScaler().fit(train_values_for_scaler.reshape(-1, 1))
        else:
            scaler = StandardScaler().fit(values.reshape(-1, 1))
        scaled = scaler.transform(values.reshape(-1, 1)).flatten().astype(np.float32)

        # Take last CONTEXT_LENGTH points as input
        if len(scaled) < CONTEXT_LENGTH:
            scaled = np.concatenate(
                [np.zeros(CONTEXT_LENGTH - len(scaled), dtype=np.float32), scaled]
            )
        context_scaled = scaled[-CONTEXT_LENGTH:]

        # Build DataFrame for pipeline
        df = pd.DataFrame({"x": context_scaled})
        df["timestamp"] = pd.date_range("2020-01-01", periods=CONTEXT_LENGTH, freq="s")

        tsp = TimeSeriesPreprocessor(
            timestamp_column="timestamp",
            id_columns=[],
            target_columns=["x"],
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            freq="s",
            scaling=False,
        )
        tsp.train(df)
        fpipe = TimeSeriesForecastingPipeline(
            self.model, feature_extractor=tsp, device=self.device
        )

        # Run inference under no_grad — avoids autograd graph overhead and
        # is safer under concurrent ThreadPoolExecutor access.
        with torch.no_grad():
            forecasts = fpipe(df)
        pred_raw = forecasts["x_prediction"].iloc[0]
        prediction_scaled = np.array(pred_raw, dtype=np.float32).flatten()[-PREDICTION_LENGTH:]

        # Inverse-transform to original scale so curves align with telemetry
        context_raw = scaler.inverse_transform(
            context_scaled.reshape(-1, 1)
        ).flatten().astype(np.float32)
        prediction_raw = scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).flatten().astype(np.float32)

        return context_raw, prediction_raw, scaler


__all__ = ["TrendForecaster", "DEFAULT_MODEL", "CONTEXT_LENGTH", "PREDICTION_LENGTH"]
