"""L3.5 rule — leak-free knee threshold on the score distribution.

Wraps :func:`phm.algorithm.persistence_filter.knee_threshold` as a
stateful :class:`BaseFilter` module so it can be registered in
:data:`FILTER_REGISTRY` and composed into an L3 / Layer-3.5 chain.

Motivation (ported from the offline validation in
``experiments/metrics/run_leakfree_metrics_v2.py`` and
``experiments/diag/eval_combined_v3.py``):
Percentile thresholds (``global_p99`` etc.) leak test-segment labels
into the threshold and collapse to 0 on near-constant channels whose
training scores are all-zero.  The Satopaa kneadle algorithm finds the
"elbow" of the sorted non-zero training-score distribution — a
genuinely leak-free threshold derived only from the known-normal
training segment.  On NASA-MSL/SMAP this lifts event-detection rate by
+16.7pp / +15.7pp compared to the leaky percentile baseline.

Two operating modes
-------------------
1. **Explicit fit (offline calibration):** call :meth:`fit` with the
   training-segment scores for a channel; the knee threshold is
   computed once and cached.  Subsequent :meth:`filter_channel` calls
   apply the cached threshold.

2. **Online accumulation:** if :meth:`fit` was never called for a
   channel, :meth:`filter_channel` accumulates the incoming scores
   into a training buffer until ``min_fit_samples`` are seen, then
   derives the knee threshold automatically.  Until the threshold is
   available the module passes scores through unchanged (warm-up).

A third shortcut — ``threshold_override`` — skips both fit and
accumulation and uses a fixed threshold directly (useful for tests and
for channels with a known operating point).

Return-value semantics
----------------------
Once the threshold is known, samples whose score is **at or below**
the threshold are zeroed (leak-free suppression of sub-threshold
noise); samples above the threshold keep their original score.  This
makes the module composable with :class:`L3PersistenceRule`, which
then binarises the surviving scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L3_PHYSICAL,
    DECISION_PASS,
)
from ..persistence_filter import knee_threshold
from ._base import register_filter

__all__ = ["L3KneeThresholdRule"]


# Default channel key used when the BaseFilter.filter() entry point is
# called without a channel id.
_DEFAULT_CHANNEL = ""

# Default score floor under which a sample is treated as zero.  Matches
# the offline scripts' convention (``scores > eps`` selects non-zero).
_DEFAULT_EPS = 1e-6

# Default minimum accumulated samples before the knee threshold is
# derived in online mode.  512 mirrors one TSPulse context block —
# enough to characterise the score distribution without over-delaying
# the warm-up.  Callers that want a different cadence can override via
# the constructor.
_DEFAULT_MIN_FIT_SAMPLES = 512


@register_filter("l3_knee_threshold")
class L3KneeThresholdRule(BaseFilter):
    """Leak-free knee threshold on per-sample anomaly scores.

    Args:
        eps: scores at or below this are treated as zero when deriving
            the knee (default ``1e-6``).
        min_fit_samples: minimum accumulated scores before the knee
            threshold is auto-derived in online mode (default ``512``).
            Ignored when ``threshold_override`` is set or after
            :meth:`fit` is called.
        threshold_override: if not None, skip fit/accumulation and use
            this fixed threshold directly.  Useful for tests and for
            channels with a known operating point.
    """

    name = "l3_knee_threshold"

    def __init__(
        self,
        *,
        eps: float = _DEFAULT_EPS,
        min_fit_samples: int = _DEFAULT_MIN_FIT_SAMPLES,
        threshold_override: float | None = None,
    ) -> None:
        self.eps = float(eps)
        self.min_fit_samples = int(min_fit_samples)
        # Fixed override → stored directly under a sentinel channel key
        # so filter_channel() finds it without any accumulation.
        self._threshold_override = (
            None if threshold_override is None else float(threshold_override)
        )
        # Per-channel state.
        #   _threshold[channel]   → cached knee threshold (once derived)
        #   _train_buf[channel]   → list of accumulated scores (online mode)
        self._threshold: dict[str, float] = {}
        self._train_buf: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Explicit offline fit
    # ------------------------------------------------------------------

    def fit(self, channel: str, train_scores: np.ndarray) -> float:
        """Compute and cache the knee threshold for ``channel``.

        Args:
            channel: channel id whose threshold to set.
            train_scores: 1-D array of scores from a known-normal
                (training) segment.  All-zero or very-short inputs fall
                back to ``eps`` — the knee algorithm requires a tail to
                find an elbow in.

        Returns:
            The computed threshold (also cached for later
            :meth:`filter_channel` calls).
        """
        scores = np.asarray(train_scores, dtype=np.float64).ravel()
        thr = float(knee_threshold(scores, eps=self.eps))
        self._threshold[channel] = thr
        # Discard any partial accumulation — fit() is authoritative.
        self._train_buf.pop(channel, None)
        return thr

    def get_threshold(self, channel: str) -> float | None:
        """Return the cached threshold for ``channel``, or None if unfitted."""
        return self._threshold.get(channel)

    # ------------------------------------------------------------------
    # Stateful streaming entry point
    # ------------------------------------------------------------------

    def filter_channel(
        self,
        channel: str,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """Apply the knee threshold to ``scores`` for ``channel``.

        If the threshold is already cached (via :meth:`fit` or online
        accumulation) the scores are filtered: samples at or below the
        threshold are zeroed, samples above keep their value.

        If the threshold is not yet available (online warm-up), the
        scores are accumulated into the training buffer and passed
        through unchanged.  When the buffer reaches
        ``min_fit_samples`` the knee threshold is derived and applied
        from this call onward.
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
            return self._result(s, threshold=None, applied=False)

        thr = self._resolve_threshold(channel, s)
        if thr is None:
            # Warm-up: threshold not yet available, pass through.
            return self._result(s, threshold=None, applied=False)

        # Apply threshold: zero sub-threshold samples (leak-free
        # suppression).  Strict > keeps the convention used by the
        # persistence binarisation downstream.
        suppressed = s <= thr
        adjusted = s.copy()
        adjusted[suppressed] = 0.0
        return self._result(adjusted, threshold=thr, applied=True)

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
        """Clear cached threshold + accumulation for one channel (None = all)."""
        if channel is None:
            self._threshold.clear()
            self._train_buf.clear()
        else:
            self._threshold.pop(channel, None)
            self._train_buf.pop(channel, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_threshold(self, channel: str, scores: np.ndarray) -> float | None:
        """Return the cached threshold, or derive it via accumulation.

        Precedence:
          1. ``threshold_override`` (constructor) — always wins.
          2. Cached threshold from a prior :meth:`fit` / accumulation.
          3. Online accumulation: append scores to the buffer; once it
             reaches ``min_fit_samples``, derive and cache the knee.
        """
        if self._threshold_override is not None:
            return self._threshold_override

        cached = self._threshold.get(channel)
        if cached is not None:
            return cached

        # Online warm-up: accumulate and maybe derive.
        buf = self._train_buf.setdefault(channel, [])
        # Extend with the current block's scores (avoid storing huge
        # arrays — once we cross min_fit_samples we derive and clear).
        buf.extend(float(x) for x in scores)
        if len(buf) >= self.min_fit_samples:
            arr = np.asarray(buf, dtype=np.float64)
            thr = float(knee_threshold(arr, eps=self.eps))
            self._threshold[channel] = thr
            self._train_buf.pop(channel, None)
            return thr
        return None

    def _result(
        self,
        adjusted: np.ndarray,
        *,
        threshold: float | None,
        applied: bool,
    ) -> LayerResult:
        """Build a LayerResult with the knee-thresholded scores.

        ``rules`` is ``["knee_threshold"]`` once the threshold has been
        applied, empty during warm-up.  The threshold value (when known)
        is recorded in ``detail`` for diagnostics and downstream
        consumers (e.g. a UI showing the operating point).
        """
        detail: dict[str, Any] = {
            "rules": ["knee_threshold"] if applied else [],
            "adjusted_scores": adjusted.astype(np.float32),
        }
        if threshold is not None:
            detail["threshold"] = threshold
        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=DECISION_PASS,
            score=0.0,
            detail=detail,
        )
