"""L3.5 rule — temporal persistence filtering of thresholded alarms.

Wraps :class:`phm.algorithm.persistence_filter.PersistenceFilter` as a
:class:`BaseFilter` module so it can be registered in
:data:`FILTER_REGISTRY` and composed into an L3 / Layer-3.5 chain.

Motivation (ported from the offline ablation in
``experiments/metrics/run_ablation_a6.py`` group A6c/A6d):
A classic PHM false-alarm source is transient score spikes that cross
the threshold for only 1-2 consecutive samples (sensor glitches, block
boundary artefacts).  The persistence filter suppresses those by
requiring at least ``K`` of the last ``W`` consecutive samples to
exceed the threshold before confirming an alarm (causal majority vote).

Validated on NASA-MSL/SMAP under the leak-free protocol:
W=8/K=4 reduces event-wise FA/h by 29% (MSL) / 52% (SMAP) at the cost
of a small event-detection-rate drop on channels with very short
(<4-sample) anomalies.

Statefulness
------------
Persistence is causal: sample ``i``'s decision depends on the previous
``W-1`` decisions.  The module delegates cross-block state to an
internal :class:`PersistenceFilter` (which keeps a per-channel ring of
recent binary predictions).  Use :meth:`filter_channel` for the
stateful streaming path; the base :meth:`filter` falls back to a
single-channel call keyed under ``""`` so the module still satisfies
the :class:`BaseFilter` contract.

Return-value semantics
----------------------
Unlike the binary :func:`apply_persistence` primitive, this module's
``adjusted_scores`` stay continuous: confirmed samples (persistence
vote passed) keep their original score, suppressed samples are zeroed.
This lets the module sit in an L3-style chain whose downstream
consumers expect continuous scores, while still expressing the
"unconfirmed alarm → suppress" intent.
"""

from __future__ import annotations

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L3_PHYSICAL,
    DECISION_PASS,
)
from ..persistence_filter import (
    DEFAULT_PERSIST_K,
    DEFAULT_PERSIST_W,
    PersistenceConfig,
    PersistenceFilter,
)
from ._base import register_filter

__all__ = ["L3PersistenceRule"]


# Default channel key used when the BaseFilter.filter() entry point is
# called without a channel id.
_DEFAULT_CHANNEL = ""


@register_filter("l3_persistence")
class L3PersistenceRule(BaseFilter):
    """W/K causal persistence filter on thresholded anomaly scores.

    Args:
        W: persistence window (number of consecutive samples considered).
        K: minimum count of threshold-crossing samples within the window
            required to confirm an alarm.  Must satisfy ``1 <= K <= W``.
        threshold: score threshold used to binarise the input scores
            before the persistence vote (default ``0.5``, matching the
            online ``ANOMALY_THRESHOLD``).  Samples with
            ``score > threshold`` count as a positive vote.
    """

    name = "l3_persistence"

    def __init__(
        self,
        *,
        W: int = DEFAULT_PERSIST_W,
        K: int = DEFAULT_PERSIST_K,
        threshold: float = 0.5,
    ) -> None:
        # PersistenceConfig validates 1 <= K <= W and W >= 1.
        self._config = PersistenceConfig(W=W, K=K)
        self._pf = PersistenceFilter(self._config)
        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # Stateful streaming entry point
    # ------------------------------------------------------------------

    def filter_channel(
        self,
        channel: str,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """Apply persistence filtering to ``scores`` for ``channel``.

        The persistence filter maintains a per-channel ring of recent
        binary predictions so the W/K vote bridges across blocks.
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

        if n == 0:
            return self._result(s, n_confirmed=0)

        # Binarise: 1 where score exceeds the threshold (strict >).
        preds = (s > self.threshold).astype(np.int32)
        # Run the stateful persistence vote (updates the per-channel
        # history inside self._pf).
        confirmed = self._pf.update(channel, preds)

        # Build adjusted_scores: keep the original score where the
        # persistence vote confirmed the alarm, zero elsewhere.  This
        # preserves continuity for downstream consumers while expressing
        # the suppression intent.
        confirmed_mask = confirmed.astype(bool)
        adjusted = np.where(confirmed_mask, s, 0.0)
        n_confirmed = int(confirmed_mask.sum())

        return self._result(adjusted, n_confirmed=n_confirmed)

    # ------------------------------------------------------------------
    # BaseFilter entry point (channel-less fallback)
    # ------------------------------------------------------------------

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """BaseFilter-compatible entry point (uses the default channel key)."""
        return self.filter_channel(_DEFAULT_CHANNEL, values, scores)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, channel: str | None = None) -> None:
        """Clear persistence history for one channel (None = all)."""
        self._pf.reset(channel)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result(self, adjusted: np.ndarray, *, n_confirmed: int) -> LayerResult:
        """Build a LayerResult with the persistence-filtered scores.

        ``rules`` is ``["persistence"]`` when any sample was confirmed
        (i.e. the filter let an alarm through), empty otherwise.  The
        confirmed-count is recorded in ``detail`` for diagnostics.
        """
        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=DECISION_PASS,
            score=0.0,
            detail={
                "rules": ["persistence"] if n_confirmed > 0 else [],
                "adjusted_scores": adjusted.astype(np.float32),
                "n_confirmed": n_confirmed,
                "W": self._config.W,
                "K": self._config.K,
            },
        )
