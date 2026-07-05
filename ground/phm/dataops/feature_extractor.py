"""Feature-extraction plugin interface (reserved).

Not implemented this iteration.  The contract below lets future feature
extractors (wavelet, statistical, frequency-domain) plug into the
algorithm layer without changing the service code that consumes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseFeatureExtractor(ABC):
    """Feature extractor plugin contract."""

    name: str = "base"

    @abstractmethod
    def extract(self, values: np.ndarray) -> np.ndarray:
        """Return a feature matrix [n_features] or [T, n_features]."""
        raise NotImplementedError


__all__ = ["BaseFeatureExtractor"]
