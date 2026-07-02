"""Forecasting module using TTM-R3 (zero-shot prediction).

This module wraps the TTM-R3 pre-trained model to provide future value
forecasting for telemetry channels. It is designed to simulate the "ground
segment" (deeper analysis with prediction capability) in the space-ground
collaborative architecture.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

# Model constants
DEFAULT_MODEL = "ibm-research/ttm-r3"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96


class TrendForecaster:
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
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        path = model_path or DEFAULT_MODEL
        if path and os.path.isdir(path):
            # Local model: load directly without get_model
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                path, prediction_filter_length=PREDICTION_LENGTH
            )
        else:
            # Online model: use get_model for automatic branch matching
            self.model = get_model(
                model_path=path,
                context_length=CONTEXT_LENGTH,
                prediction_length=PREDICTION_LENGTH,
            )
        self.model = self.model.to(device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.model_source = path

    def forecast(self, values, train_values_for_scaler=None):
        """Forecast future PREDICTION_LENGTH steps from the last CONTEXT_LENGTH steps.

        Args:
            values: np.ndarray [T] float32 — telemetry values (T >= CONTEXT_LENGTH)
            train_values_for_scaler: np.ndarray or None — for StandardScaler fitting

        Returns:
            context: np.ndarray [CONTEXT_LENGTH] — the input window (standardized)
            prediction: np.ndarray [PREDICTION_LENGTH] — forecasted values (standardized)
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
        context = scaled[-CONTEXT_LENGTH:]

        # Build DataFrame for pipeline
        df = pd.DataFrame({"x": context})
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

        forecasts = fpipe(df)
        pred_raw = forecasts["x_prediction"].iloc[0]
        prediction = np.array(pred_raw, dtype=np.float32).flatten()[-PREDICTION_LENGTH:]

        return context, prediction
