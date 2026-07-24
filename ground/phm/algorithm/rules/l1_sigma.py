"""L1 rule — 3σ outlier detection.

Ported verbatim from the σ check of ``ClassicFilter.filter``
(``classic_filter.py:151-167``).  Any finite sample beyond
``mean ± sigma_k·std`` is flagged; flagged samples get a per-sample
score of ``0.8`` and the rule contributes rule name ``"sigma_3"`` to
the merged detail.

Critical operator-precedence note (preserved from the original):
``finite_mask & ((v < lo) | (v > hi))`` — the inner parentheses are
*required* because ``&`` binds tighter than ``|`` in Python.  Writing
``finite_mask & (v < lo) | (v > hi)`` would parse as
``(finite_mask & (v < lo)) | (v > hi)`` and incorrectly flag ``+Inf``
samples (``v > hi`` is True for ``+Inf`` even though ``finite_mask`` is
False).  Non-finite samples are sanitised downstream by L3, not L1.
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

__all__ = ["L1SigmaRule"]


# Score assigned to σ outliers — matches the original magic number in
# ClassicFilter (``per_sample[sigma_out] = np.maximum(..., 0.8)``).
SIGMA_OUTLIER_SCORE = 0.8


@register_filter("l1_sigma")
class L1SigmaRule(BaseFilter):
    """Flag samples beyond ``mean ± sigma_k·std`` (3σ rule).

    Args:
        sigma_k: number of standard deviations (default ``3.0``).
        min_sigma: σ is only meaningfully evaluated when ``sigma > 0``;
            this is preserved from the original guard so degenerate
            near-constant windows (handled by ``l1_constant``) don't
            divide by ~0.
    """

    name = "l1_sigma"

    def __init__(self, *, sigma_k: float = 3.0, min_sigma: float = 0.0) -> None:
        self.sigma_k = float(sigma_k)
        self.min_sigma = float(min_sigma)

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
        if n_finite < 2:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=decision,
                score=0.0,
                detail={"rules": rules, "per_sample_score": per_sample},
            )

        clean = v[finite_mask]
        mu = float(np.mean(clean))
        sigma = float(np.std(clean))

        if sigma > self.min_sigma:
            lo = mu - self.sigma_k * sigma
            hi = mu + self.sigma_k * sigma
            # NOTE: parentheses required — see module docstring.
            sigma_out = finite_mask & ((v < lo) | (v > hi))
            n_sigma = int(sigma_out.sum())
            if n_sigma > 0:
                rules.append("sigma_3")
                per_sample[sigma_out] = np.maximum(per_sample[sigma_out], SIGMA_OUTLIER_SCORE)
                decision = DECISION_ALERT

        rep_score = float(np.max(per_sample)) if n > 0 else 0.0
        return LayerResult(
            layer=LAYER_L1_CLASSIC,
            decision=decision,
            score=rep_score,
            detail={
                "rules": rules,
                "mean": mu,
                "std": sigma,
                "per_sample_score": per_sample,
            },
        )
