"""Algorithm layer entry-point.

Re-exports the active detector / forecaster / cascade components so callers
can do::

    from phm.algorithm import AnomalyDetector, TrendForecaster
    from phm.algorithm import CascadeDetector, ClassicFilter, PhysicalConstraint

without knowing the concrete module name.  This also keeps the future
model registry swap-point localised.
"""

from .base import BaseDetector, BaseForecaster, BaseRULPredictor
from ._registry import MODEL_REGISTRY, ModelEntry, get_model_entry
from .tspulse import AnomalyDetector, DEFAULT_MODEL as TSPULSE_DEFAULT_MODEL
from .ttm import TrendForecaster
from .ttm import DEFAULT_MODEL as TTM_DEFAULT_MODEL
from .ttm import CONTEXT_LENGTH, PREDICTION_LENGTH
from .rul_model import RULPredictor

# Cascade components
from .base_filter import BaseFilter
from .cascade_types import LayerResult, CascadeOutput
from .classic_filter import ClassicFilter
from .physical_constraint import ConstraintConfig, PhysicalConstraint
from .cascade_detector import CascadeDetector
# Calibration components (offline-tuned enhancements)
from .direction_calibrator import DirectionCalibrator
from .freq_feature import FreqFeatureExtractor
from .calibration_config import CalibrationConfig, ChannelCalibration
# Leak-free post-processing (knee threshold + EMA smoothing + persistence).
# Optional Layer 3.5 — off by default, validated in
# experiments/metrics/run_ablation_a6.py.
from .persistence_filter import (
    DEFAULT_EMA_ALPHA,
    DEFAULT_PERSIST_K,
    DEFAULT_PERSIST_W,
    PersistenceConfig,
    PersistenceFilter,
    apply_persistence,
    causal_ema,
    knee_threshold,
)

__all__ = [
    # Base plugin contracts
    "BaseDetector",
    "BaseForecaster",
    "BaseRULPredictor",
    "BaseFilter",
    # Concrete implementations
    "AnomalyDetector",
    "TrendForecaster",
    "RULPredictor",
    "CascadeDetector",
    "ClassicFilter",
    "PhysicalConstraint",
    "ConstraintConfig",
    # Calibration components
    "DirectionCalibrator",
    "FreqFeatureExtractor",
    "CalibrationConfig",
    "ChannelCalibration",
    # Leak-free post-processing (Layer 3.5, optional)
    "PersistenceConfig",
    "PersistenceFilter",
    "apply_persistence",
    "knee_threshold",
    "causal_ema",
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_PERSIST_W",
    "DEFAULT_PERSIST_K",
    # Cascade data types
    "LayerResult",
    "CascadeOutput",
    # Model registry (agent-friendly: enumerate models without torch import)
    "MODEL_REGISTRY",
    "ModelEntry",
    "get_model_entry",
    # Model constants (backwards-compat aliases)
    "TSPULSE_DEFAULT_MODEL",
    "TTM_DEFAULT_MODEL",
    "CONTEXT_LENGTH",
    "PREDICTION_LENGTH",
]
