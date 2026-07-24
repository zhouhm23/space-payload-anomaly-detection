"""L3 rule — rate-of-change ceiling boost.

Ported verbatim from rule 4 of ``PhysicalConstraint.filter``
(``physical_constraint.py:171-179``).  Consecutive-sample jumps
exceeding ``max_rate`` are physically unreasonable ⇒ their score is
boosted toward ``rate_boost`` (default ``0.85``).  Rule name
contributed: ``"rate_ceiling"``.

Disabled when ``max_rate`` is None (the original default).
"""

from __future__ import annotations

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L3_PHYSICAL,
    DECISION_PASS,
)
from ._base import register_filter

__all__ = ["L3RateRule"]


# Minimum sample count for diff to be meaningful — matches original
# guard ``n >= 2``.
RATE_MIN_SAMPLES = 2


@register_filter("l3_rate")
class L3RateRule(BaseFilter):
    """Boost scores for samples whose jump exceeds ``max_rate``.

    Args:
        max_rate: rate-of-change ceiling (None disables the rule).
        rate_boost: target score for samples whose preceding jump
            exceeds ``max_rate`` (default ``0.85``).
    """

    name = "l3_rate"

    def __init__(
        self,
        *,
        max_rate: float | None = None,
        rate_boost: float = 0.85,
    ) -> None:
        self.max_rate = None if max_rate is None else float(max_rate)
        self.rate_boost = float(rate_boost)

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)
        if scores is None:
            s = np.zeros(n, dtype=np.float64)
        else:
            s = np.asarray(scores, dtype=np.float64).ravel().copy()
            if len(s) != n:
                if len(s) < n:
                    s = np.concatenate([s, np.zeros(n - len(s))])
                else:
                    s = s[:n]

        rules: list[str] = []
        if self.max_rate is not None and n >= RATE_MIN_SAMPLES:
            diffs = np.abs(np.diff(v))
            rate_out = np.zeros(n, dtype=bool)
            rate_out[1:] = diffs > self.max_rate
            n_rate = int(rate_out.sum())
            if n_rate > 0:
                s[rate_out] = np.maximum(s[rate_out], self.rate_boost)
                rules.append("rate_ceiling")

        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=DECISION_PASS,
            score=0.0,
            detail={
                "rules": rules,
                "adjusted_scores": s.astype(np.float32),
            },
        )
