"""L1 rule — constant / near-constant channel detection + data-quality guard.

Ported verbatim (logic + thresholds) from the first section of
``ClassicFilter.filter`` (``classic_filter.py:105-145``).  This rule owns
two distinct checks that were adjacent in the original monolithic filter:

1. **Empty input** (``n == 0``) — always returns ``skip`` with reason
   ``empty_input``.  Not gated by any enable flag.
2. **Constant-channel detection** (``std < constant_std``) — gated by
   ``enable_constant``; returns ``skip`` with rule ``constant_channel``.
3. **Insufficient-finite guard** (``n_finite < min_finite``) — always
   returns ``skip`` with rule ``insufficient_finite``.  Not gated by
   ``enable_constant`` (matches the original: this check sits *outside* the
   ``enable_constant`` guard so even with the std check disabled the
   cascade still short-circuits on unusable data).

When none of the above triggers, the rule returns ``pass`` with an
all-zero ``per_sample_score`` so the combiner (:class:`ClassicFilter`)
can merge it with the other L1 rules' contributions.
"""

from __future__ import annotations

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L1_CLASSIC,
    DECISION_PASS,
    DECISION_SKIP,
)
from ._base import register_filter

__all__ = ["L1ConstantRule"]


@register_filter("l1_constant")
class L1ConstantRule(BaseFilter):
    """Detect near-constant channels and guard against unusable input.

    Args:
        constant_std: channels with finite-sample std below this are
            treated as constant (default ``1e-3``, matching the original
            ClassicFilter default).
        min_finite: minimum finite-sample count required to evaluate the
            std rule and to consider the block usable (default ``2``).
        enable_constant: when ``False`` the std check is skipped but the
            empty-input and insufficient-finite guards still run (matches
            the original control flow where only the std check was gated
            by ``enable_constant``).
    """

    name = "l1_constant"

    def __init__(
        self,
        *,
        constant_std: float = 1e-3,
        min_finite: int = 2,
        enable_constant: bool = True,
    ) -> None:
        self.constant_std = float(constant_std)
        self.min_finite = int(min_finite)
        self.enable_constant = bool(enable_constant)

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)

        # --- empty input → short-circuit (original n==0 branch) -----------
        if n == 0:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=DECISION_SKIP,
                score=0.0,
                detail={"reason": "empty_input"},
            )

        finite_mask = np.isfinite(v)
        n_finite = int(finite_mask.sum())

        # --- constant-channel std check (gated by enable_constant) -------
        if self.enable_constant and n_finite >= self.min_finite:
            finite_vals = v[finite_mask]
            std = float(np.std(finite_vals))
            if std < self.constant_std:
                return LayerResult(
                    layer=LAYER_L1_CLASSIC,
                    decision=DECISION_SKIP,
                    score=0.0,
                    detail={
                        "rules": ["constant_channel"],
                        "std": std,
                        "threshold": self.constant_std,
                        "n_samples": n,
                    },
                )

        # --- insufficient-finite guard (always on) -----------------------
        if n_finite < self.min_finite:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=DECISION_SKIP,
                score=0.0,
                detail={
                    "rules": ["insufficient_finite"],
                    "n_finite": n_finite,
                    "n_samples": n,
                },
            )

        # --- no trigger --------------------------------------------------
        return LayerResult(
            layer=LAYER_L1_CLASSIC,
            decision=DECISION_PASS,
            score=0.0,
            detail={
                "rules": [],
                "per_sample_score": np.zeros(n, dtype=np.float32),
            },
        )
