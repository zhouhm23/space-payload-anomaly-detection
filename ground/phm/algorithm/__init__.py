"""Algorithm layer entry-point.

Re-exports the active detector / forecaster so callers can do::

    from phm.algorithm import AnomalyDetector, TrendForecaster

without knowing the concrete module name.  This also keeps the future
model registry swap-point localised.
"""

from .base import BaseDetector, BaseForecaster
from .tspulse import AnomalyDetector, DEFAULT_MODEL as TSPULSE_DEFAULT_MODEL
from .ttm import TrendForecaster
from .ttm import DEFAULT_MODEL as TTM_DEFAULT_MODEL
from .ttm import CONTEXT_LENGTH, PREDICTION_LENGTH

__all__ = [
    "BaseDetector",
    "BaseForecaster",
    "AnomalyDetector",
    "TrendForecaster",
    "TSPULSE_DEFAULT_MODEL",
    "TTM_DEFAULT_MODEL",
    "CONTEXT_LENGTH",
    "PREDICTION_LENGTH",
]
