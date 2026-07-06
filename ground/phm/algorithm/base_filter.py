"""Base class for cascade filter plugins (Layer 1 & Layer 3).

Mirrors the ``BaseDetector`` / ``BaseForecaster`` plugin pattern so new
filter implementations can be dropped in without changing the cascade
orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .cascade_types import LayerResult


class BaseFilter(ABC):
    """Cascade filter plugin contract.

    A filter inspects the raw telemetry block (and optionally the upstream
    per-sample anomaly scores) and returns a :class:`LayerResult` describing
    its decision.

    Layer 1 filters (classic algorithms) run *before* the DL detector.  They
    receive only ``values`` (``scores`` is ``None``).

    Layer 3 filters (physical constraints) run *after* the DL detector.  They
    receive both ``values`` and ``scores`` (the raw DL output) and may adjust
    the scores.
    """

    name: str = "base_filter"

    @abstractmethod
    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """Evaluate this filter on one data block.

        Args:
            values: 1-D float telemetry array ``[T]``.
            scores: per-sample anomaly scores from the previous layer, or
                    ``None`` if this filter runs before the DL detector.

        Returns:
            LayerResult with a decision and optional adjusted scores.
        """
        raise NotImplementedError


__all__ = ["BaseFilter"]
