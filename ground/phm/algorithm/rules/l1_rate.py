"""L1 rule — rate-of-change (jump) detection.

Ported verbatim from the rate check of ``ClassicFilter.filter``
(``classic_filter.py:188-206``).  Flags samples whose consecutive-sample
absolute difference exceeds ``max_rate`` (or, when ``max_rate`` is None,
``rate_quantile``-th percentile of finite diffs × ``rate_multiplier``).

Flagged samples get a per-sample score of ``0.6`` (the lowest of the four
L1 scores — matches the original severity ordering σ > IQR > rate).  The
rule contributes rule name ``"rate_of_change"`` and lowers the merged
decision to ``suspicious`` (not ``alert``) — see the combinator in
:class:`ClassicFilter` for how this is folded into the final L1 decision.
"""

from __future__ import annotations

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L1_CLASSIC,
    DECISION_PASS,
    DECISION_SUSPICIOUS,
)
from ._base import register_filter

__all__ = ["L1RateRule"]


# Score assigned to rate outliers — matches the original magic number
# (``per_sample[rate_out] = np.maximum(..., 0.6)``).
RATE_OUTLIER_SCORE = 0.6
# Minimum sample count for the diff to be meaningful — matches the
# original guard ``n >= 2``.
RATE_MIN_SAMPLES = 2


@register_filter("l1_rate")
class L1RateRule(BaseFilter):
    """Flag samples whose consecutive-sample jump exceeds a threshold.

    Args:
        rate_quantile: if ``max_rate`` is None, the threshold is derived
            from the data as this percentile of absolute finite diffs
            (default ``99.0`` → only extreme jumps trigger).
        rate_multiplier: ``max_rate = percentile(rate_quantile) *
            rate_multiplier`` (default ``5.0``).
        max_rate: explicit rate-of-change threshold.  Overrides the
            quantile derivation when not None.
    """

    name = "l1_rate"

    def __init__(
        self,
        *,
        rate_quantile: float = 99.0,
        rate_multiplier: float = 5.0,
        max_rate: float | None = None,
    ) -> None:
        self.rate_quantile = float(rate_quantile)
        self.rate_multiplier = float(rate_multiplier)
        self.max_rate = None if max_rate is None else float(max_rate)

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

        if n < RATE_MIN_SAMPLES:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=decision,
                score=0.0,
                detail={"rules": rules, "per_sample_score": per_sample},
            )

        diffs = np.abs(np.diff(v))
        finite_diffs = diffs[np.isfinite(diffs)]
        if len(finite_diffs) == 0:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=decision,
                score=0.0,
                detail={"rules": rules, "per_sample_score": per_sample},
            )

        if self.max_rate is not None:
            thr = self.max_rate
        else:
            p = float(np.percentile(finite_diffs, self.rate_quantile))
            thr = p * self.rate_multiplier

        if thr > 0:
            rate_out = np.zeros(n, dtype=bool)
            rate_out[1:] = diffs > thr
            n_rate = int(rate_out.sum())
            if n_rate > 0:
                rules.append("rate_of_change")
                per_sample[rate_out] = np.maximum(per_sample[rate_out], RATE_OUTLIER_SCORE)
                decision = DECISION_SUSPICIOUS

        rep_score = float(np.max(per_sample)) if n > 0 else 0.0
        return LayerResult(
            layer=LAYER_L1_CLASSIC,
            decision=decision,
            score=rep_score,
            detail={"rules": rules, "per_sample_score": per_sample},
        )
