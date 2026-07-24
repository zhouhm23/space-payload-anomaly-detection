"""L3 rule — value-range boundary boost.

Ported verbatim from rule 3 of ``PhysicalConstraint.filter``
(``physical_constraint.py:160-169``).  Values outside
``[valid_min, valid_max]`` are physically impossible ⇒ their score is
boosted toward ``range_boost`` (default ``0.95``) regardless of what
the DL detector said.  Rule name contributed: ``"range_boundary"``.

Both bounds are optional (``None`` = unbounded on that side), matching
the original config defaults — when both are ``None`` the rule is a
no-op pass-through.
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

__all__ = ["L3RangeRule"]


@register_filter("l3_range")
class L3RangeRule(BaseFilter):
    """Boost scores for samples outside a valid value range.

    Args:
        valid_min: lower physical bound (None = no lower bound).
        valid_max: upper physical bound (None = no upper bound).
        range_boost: target score for out-of-range samples
            (default ``0.95``).
    """

    name = "l3_range"

    def __init__(
        self,
        *,
        valid_min: float | None = None,
        valid_max: float | None = None,
        range_boost: float = 0.95,
    ) -> None:
        self.valid_min = None if valid_min is None else float(valid_min)
        self.valid_max = None if valid_max is None else float(valid_max)
        self.range_boost = float(range_boost)

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
        finite_mask = np.isfinite(v)

        boosted_range = np.zeros(n, dtype=bool)
        if self.valid_min is not None:
            boosted_range |= finite_mask & (v < self.valid_min)
        if self.valid_max is not None:
            boosted_range |= finite_mask & (v > self.valid_max)
        n_range = int(boosted_range.sum())
        if n_range > 0:
            s[boosted_range] = np.maximum(s[boosted_range], self.range_boost)
            rules.append("range_boundary")

        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=DECISION_PASS,
            score=0.0,
            detail={
                "rules": rules,
                "adjusted_scores": s.astype(np.float32),
            },
        )
