"""Algorithm layer: unified plugin base classes.

Defining ``BaseDetector`` and ``BaseForecaster`` lets future models
(e.g. MOMENT, fine-tuned variants) plug into the PHM services without the
services needing to know the concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Anomaly detector plugin contract.

    Implementations score each sample of a 1-D telemetry array; higher
    scores mean more anomalous.  Examples: TSPulse.
    """

    n_params: int = 0
    model_source: str = ""

    @abstractmethod
    def detect(
        self,
        values: np.ndarray,
        train_values_for_scaler: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return per-sample anomaly scores (len == len(values))."""
        raise NotImplementedError


class BaseForecaster(ABC):
    """Time-series forecaster plugin contract.

    Implementations predict ``PREDICTION_LENGTH`` future steps from the last
    ``CONTEXT_LENGTH`` observed steps.  Examples: TTM-R3.
    """

    n_params: int = 0
    model_source: str = ""

    @abstractmethod
    def forecast(
        self,
        values: np.ndarray,
        train_values_for_scaler: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Return (context_raw, prediction_raw, scaler)."""
        raise NotImplementedError


class BaseRULPredictor(ABC):
    """Remaining-Useful-Life regression plugin contract.

    Implementations take a multivariate sensor window and predict a scalar
    RUL value (remaining cycles).  Examples: LSTM+Attention trained on
    C-MAPSS.
    """

    n_params: int = 0
    model_source: str = ""

    @abstractmethod
    def predict_rul(self, window: np.ndarray) -> float:
        """Predict RUL (remaining cycles) from a sensor window.

        Args:
            window: ``(window_size, n_sensors)`` normalised sensor readings.

        Returns:
            Predicted remaining useful life in cycles (float, ≥ 0).
        """
        raise NotImplementedError


__all__ = ["BaseDetector", "BaseForecaster", "BaseRULPredictor"]
