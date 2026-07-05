"""Ground-side preprocessor.

Thin wrapper around ``space.preprocessing.SpacePreprocessor`` so the ground
forecast+detect pipeline can reuse the exact same imputation +
standardisation logic that the space segment applies.  Keeping a separate
symbol here (rather than importing the space class directly) lets a future
ground-specific preprocessing strategy override it without touching space.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Reuse the space-segment preprocessor verbatim.  The space package lives
# one level up from ``phm`` — resolve it relative to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPACE_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "space"))
if _SPACE_DIR not in sys.path:
    sys.path.insert(0, _SPACE_DIR)

from preprocessing import SpacePreprocessor  # type: ignore  # noqa: E402


class GroundPreprocessor(SpacePreprocessor):
    """Ground-segment preprocessor — identical behaviour to space.

    Subclassed (rather than aliased) so future ground-only steps can be
    layered in without modifying the shared space implementation.
    """


__all__ = ["GroundPreprocessor"]
