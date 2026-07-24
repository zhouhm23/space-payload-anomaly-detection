"""L3.5 rule — causal EMA smoothing of anomaly scores.

Wraps :func:`phm.algorithm.persistence_filter.causal_ema` as a stateful
:class:`BaseFilter` module so it can be registered in
:data:`FILTER_REGISTRY` and composed into an L3 / Layer-3.5 chain.

Motivation (ported from the offline validation in
``experiments/diag/eval_score_smoothing.py``):
Each 512-sample block is scored independently by TSPulse with a pinned
seed, which produces visible discontinuities at block boundaries.  A
causal EMA ``y[i] = α·x[i] + (1-α)·y[i-1]`` applied to the score stream
removes those seams and improves the threshold-free VUS-PR metric by
+7.9% on NASA-MSL / +7.7% on NASA-SMAP at α=0.2, with no per-channel
regressions.

Statefulness
------------
EMA is inherently causal: each output depends on the previous output.
To bridge across streaming blocks the module keeps a per-channel
``y_last`` scalar (seeded at 0) and resumes the recursion from it on
every call.  Use :meth:`filter_channel` for the stateful streaming
path; the base :meth:`filter` falls back to a single-channel
stateless-style call keyed under ``""`` so the module still satisfies
the :class:`BaseFilter` contract when composed without a channel id.
"""

from __future__ import annotations

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L3_PHYSICAL,
    DECISION_PASS,
)
from ..persistence_filter import DEFAULT_EMA_ALPHA, causal_ema
from ._base import register_filter

__all__ = ["L3EmaSmoothingRule"]


# Default channel key used when the BaseFilter.filter() entry point is
# called without a channel id (keeps the state dict keyed consistently).
_DEFAULT_CHANNEL = ""


@register_filter("l3_ema_smoothing")
class L3EmaSmoothingRule(BaseFilter):
    """Causal EMA smoothing of per-sample anomaly scores.

    Args:
        alpha: EMA smoothing factor in ``(0, 1]``.  ``alpha=1.0`` is the
            identity (no smoothing).  Smaller α = heavier smoothing but
            more lag.  Default ``0.2`` (validated on MSL/SMAP for peak
            VUS-PR gain with no regressions).
    """

    name = "l3_ema_smoothing"

    def __init__(self, *, alpha: float = DEFAULT_EMA_ALPHA) -> None:
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        # Per-channel ``y_last`` seed.  Seeded at 0 (matches causal_ema's
        # ``y[-1] = 0`` convention) so the first sample of the very first
        # block starts from 0.
        self._y_last: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Stateful streaming entry point
    # ------------------------------------------------------------------

    def filter_channel(
        self,
        channel: str,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """EMA-smooth ``scores`` for ``channel``, carrying ``y_last`` across calls.

        This is the streaming path used by online callers that process
        one block at a time: the last smoothed value is stored under
        ``channel`` and used as the seed for the next block so the EMA
        recursion is continuous across block boundaries.
        """
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

        if self.alpha >= 1.0 or n == 0:
            # Identity / empty — no state change, pass scores through.
            return self._result(s, smoothed=False)

        # Resume the causal EMA from the stored seed.  We inline the
        # recursion (rather than calling causal_ema) so we can carry the
        # scalar y_last across calls without re-fitting a full-series
        # function each block.
        y = self._y_last.get(channel, 0.0)
        a = self.alpha
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            y = a * float(s[i]) + (1.0 - a) * y
            out[i] = y
        # Persist the final value as the seed for the next block.
        self._y_last[channel] = float(out[-1])

        return self._result(out, smoothed=True)

    # ------------------------------------------------------------------
    # BaseFilter entry point (channel-less fallback)
    # ------------------------------------------------------------------

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """BaseFilter-compatible entry point.

        Uses the default channel key ``""`` so the module can be placed
        in a :class:`PhysicalConstraint` rule chain (which calls
        ``rule.filter(values, scores)`` without a channel id).  When
        composed this way the EMA state is shared under the empty key —
        fine for single-channel offline evaluation, but for true
        multi-channel streaming use :meth:`filter_channel`.
        """
        return self.filter_channel(_DEFAULT_CHANNEL, values, scores)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, channel: str | None = None) -> None:
        """Clear the EMA seed for one channel (None = all channels)."""
        if channel is None:
            self._y_last.clear()
        else:
            self._y_last.pop(channel, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result(self, adjusted: np.ndarray, *, smoothed: bool) -> LayerResult:
        """Build a LayerResult with the EMA-smoothed adjusted_scores.

        ``rules`` is ``["ema_smoothing"]`` when smoothing was actually
        applied, and empty when α=1.0 (identity) or the block was empty —
        matching the convention of the other L3 modules where ``rules``
        records what fired.
        """
        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=DECISION_PASS,
            score=0.0,
            detail={
                "rules": ["ema_smoothing"] if smoothed else [],
                "adjusted_scores": adjusted.astype(np.float32),
                "alpha": self.alpha,
            },
        )
