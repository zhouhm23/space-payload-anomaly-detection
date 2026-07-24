"""L3 rule — constant-channel suppression.

Ported verbatim from rule 2 of ``PhysicalConstraint.filter``
(``physical_constraint.py:139-158``).  When the input window's
finite-sample std is below ``constant_std``, all scores are forced to
``0.0``.  This eliminates the ``threshold=0 → recall=1.0`` false
positive that Day-8 analysis revealed on near-constant TSB-UAD
channels (C-2/D-14/M-6/S-2/T-5).

In the original code this rule has an **early return**: when it
triggers the L3 result is returned immediately, skipping rules 3/4/5.
The :class:`PhysicalConstraint` combinator preserves that by stopping
the chain after this rule returns ``triggered=True`` in its detail.
Rule name contributed: ``"constant_suppression"``.
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

__all__ = ["L3ConstantRule"]


@register_filter("l3_constant")
class L3ConstantRule(BaseFilter):
    """Force all scores to 0 when the input window is near-constant.

    Args:
        constant_std: windows with finite-sample std below this are
            suppressed (default ``1e-3``).
        min_finite: minimum finite-sample count to evaluate std
            (default ``2``).
    """

    name = "l3_constant"

    def __init__(
        self,
        *,
        constant_std: float = 1e-3,
        min_finite: int = 2,
    ) -> None:
        self.constant_std = float(constant_std)
        self.min_finite = int(min_finite)

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
        n_finite = int(finite_mask.sum())

        if n_finite >= self.min_finite:
            clean = v[finite_mask]
            std = float(np.std(clean))
            if std < self.constant_std:
                s[:] = 0.0
                rules.append("constant_suppression")
                return LayerResult(
                    layer=LAYER_L3_PHYSICAL,
                    decision=DECISION_PASS,
                    score=0.0,
                    detail={
                        "rules": rules,
                        "adjusted_scores": s.astype(np.float32),
                        "std": std,
                        # Signal to the PhysicalConstraint combinator that
                        # the original code early-returns here (rules 3/4/5
                        # are skipped).  Stage-1 does not change behaviour,
                        # so the combinator stops the chain when this flag
                        # is present.
                        "_stop_chain": True,
                    },
                )

        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=DECISION_PASS,
            score=0.0,
            detail={
                "rules": rules,
                "adjusted_scores": s.astype(np.float32),
            },
        )
