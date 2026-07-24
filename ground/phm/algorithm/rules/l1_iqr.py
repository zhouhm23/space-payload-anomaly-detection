"""L1 rule — IQR (interquartile range) outlier detection.

Ported verbatim from the IQR check of ``ClassicFilter.filter``
(``classic_filter.py:169-186``).  Any finite sample outside
``[Q1 − iqr_factor·IQR, Q3 + iqr_factor·IQR]`` is flagged; flagged
samples get a per-sample score of ``0.7`` and the rule contributes
rule name ``"iqr"`` to the merged detail.

Operator-precedence note identical to :mod:`l1_sigma`: the inner
parentheses in ``finite_mask & ((v < lo_iqr) | (v > hi_iqr))`` are
required so non-finite samples are excluded before the comparisons OR.
"""

from __future__ import annotations

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L1_CLASSIC,
    DECISION_ALERT,
    DECISION_PASS,
)
from ._base import register_filter

__all__ = ["L1IqrRule"]


# Score assigned to IQR outliers — matches the original magic number
# (``per_sample[iqr_out] = np.maximum(..., 0.7)``).
IQR_OUTLIER_SCORE = 0.7
# Minimum finite-sample count for the IQR to be meaningful — matches the
# original guard ``n_finite >= 4`` (need at least 4 points for quartiles).
IQR_MIN_FINITE = 4


@register_filter("l1_iqr")
class L1IqrRule(BaseFilter):
    """Flag samples outside the IQR fence (Tukey rule).

    Args:
        iqr_factor: multiplier for the IQR fence (default ``1.5``,
            the classic Tukey value).
        min_finite: minimum finite-sample count required for quartile
            estimation (default ``4``).
    """

    name = "l1_iqr"

    def __init__(
        self,
        *,
        iqr_factor: float = 1.5,
        min_finite: int = IQR_MIN_FINITE,
    ) -> None:
        self.iqr_factor = float(iqr_factor)
        self.min_finite = int(min_finite)

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)

        per_sample = np.zeros(n, dtype=np.float32)
        rules: list[str] = []
        decision = DECISION_PASS

        if n == 0:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=decision,
                score=0.0,
                detail={"rules": rules, "per_sample_score": per_sample},
            )

        finite_mask = np.isfinite(v)
        n_finite = int(finite_mask.sum())
        if n_finite < self.min_finite:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=decision,
                score=0.0,
                detail={"rules": rules, "per_sample_score": per_sample},
            )

        clean = v[finite_mask]
        q1 = float(np.percentile(clean, 25))
        q3 = float(np.percentile(clean, 75))
        iqr = q3 - q1
        if iqr > 0:
            lo_iqr = q1 - self.iqr_factor * iqr
            hi_iqr = q3 + self.iqr_factor * iqr
            # NOTE: parentheses required — see module docstring.
            iqr_out = finite_mask & ((v < lo_iqr) | (v > hi_iqr))
            n_iqr = int(iqr_out.sum())
            if n_iqr > 0:
                rules.append("iqr")
                per_sample[iqr_out] = np.maximum(per_sample[iqr_out], IQR_OUTLIER_SCORE)
                decision = DECISION_ALERT

        rep_score = float(np.max(per_sample)) if n > 0 else 0.0
        return LayerResult(
            layer=LAYER_L1_CLASSIC,
            decision=decision,
            score=rep_score,
            detail={"rules": rules, "per_sample_score": per_sample},
        )
