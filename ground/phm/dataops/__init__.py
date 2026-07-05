"""Data operations layer.

Re-uses the space-segment preprocessing pipeline (impute + standardise) on
the ground side for the forecast+detect combined pipeline, and reserves a
feature-extraction plugin interface for future extension.
"""

from .preprocessor import GroundPreprocessor

__all__ = ["GroundPreprocessor"]
