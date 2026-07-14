"""Algorithm layer entry-point.

Re-exports the active detector / forecaster / cascade components so callers
can do::

    from phm.algorithm import AnomalyDetector, TrendForecaster
    from phm.algorithm import CascadeDetector, ClassicFilter, PhysicalConstraint

without knowing the concrete module name.  This also keeps the future
model registry swap-point localised.
"""

from .base import BaseDetector, BaseForecaster, BaseRULPredictor
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
    # Cascade data types
    "LayerResult",
    "CascadeOutput",
    # Model constants
    "TSPULSE_DEFAULT_MODEL",
    "TTM_DEFAULT_MODEL",
    "CONTEXT_LENGTH",
    "PREDICTION_LENGTH",
]
