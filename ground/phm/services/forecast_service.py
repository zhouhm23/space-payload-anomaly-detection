"""Forecast service — TTM-R3 with linear-extrapolation fallback.

Lazily instantiates the TTM-R3 forecaster (heavy model load) on first use
and caches it.  Preserves the legacy fallback chain: try TTM-R3 first, on
any failure fall back to a slope-based linear extrapolation so the API
never 500s just because the model is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np

from ..algorithm import BaseForecaster

logger = logging.getLogger(__name__)


class ForecastService:
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._forecaster: BaseForecaster | None = None
        self._init_failed = False

    def _get_forecaster(self) -> BaseForecaster | None:
        if self._forecaster is not None:
            return self._forecaster
        if self._init_failed:
            return None
        try:
            from ..algorithm import TrendForecaster
            self._forecaster = TrendForecaster(device=self._device)
        except Exception as e:
            logger.warning("Failed to load TTM-R3 forecaster: %s", e)
            self._init_failed = True
            return None
        return self._forecaster

    def forecast(self, values: list[float]) -> dict:
        """Forecast 96 future steps.

        Returns ``{"context", "prediction", "model"}`` — same shape as the
        legacy ``api_forecast`` response.
        """
        if len(values) < 10:
            return {"error": "Need at least 10 data points"}

        arr = np.array(values, dtype=np.float32)

        fc = self._get_forecaster()
        if fc is not None:
            try:
                context, prediction, _ = fc.forecast(arr)
                return {
                    "context": context.tolist(),
                    "prediction": prediction.tolist(),
                    "model": "ttm-r3",
                }
            except Exception as e:
                logger.warning("TTM-R3 forecast failed, falling back to linear: %s", e)

        # Linear fallback — identical to legacy implementation
        n = min(96, len(arr))
        recent = arr[-n:]
        x = np.arange(n, dtype=np.float64)
        y = recent.astype(np.float64)
        slope = np.polyfit(x, y, 1)[0]
        last_val = float(arr[-1])
        prediction = [last_val + slope * (i + 1) for i in range(96)]
        context = arr[-min(512, len(arr)):].tolist()
        return {
            "context": context,
            "prediction": prediction,
            "model": "linear",
        }


__all__ = ["ForecastService"]
