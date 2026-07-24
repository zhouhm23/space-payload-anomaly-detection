"""Layer 3.5 — Persistence post-filter (temporal smoothing of alarms).

Runs *after* L3 physical constraints.  Suppresses transient spikes that pass
the threshold for only 1-2 consecutive samples — a classic PHM false-alarm
source.  An alarm is confirmed only when at least ``K`` of the last ``W``
consecutive samples exceed the threshold (causal majority vote).

Motivation
----------
The leak-free v1 baseline (``run_leakfree_metrics.py``) suffers from two
false-alarm sources that this filter addresses:

1. **Threshold collapse on near-constant channels** — six NASA-MSL channels
   (C-2/D-14/M-6/P-14/S-2/T-5) have TSPulse + L3 constant-suppression
   producing *all-zero* scores on the all-normal train segment.  Any
   leak-free percentile threshold collapses to 0, and every positive test
   score becomes a false alarm.  The persistence filter removes the
   resulting transient spikes because real anomalies persist; sensor
   glitches do not.

2. **Score-distribution overlap** — on channels whose anomaly and normal
   score distributions overlap (F-7, T-12, ...), the threshold necessarily
   sits in a noisy region.  Persistence removes isolated threshold
   crossings while preserving sustained anomalies.

Validated in ``experiments/metrics/run_ablation_a6.py`` (group A6c, A6d)
on NASA-MSL and NASA-SMAP under the leak-free protocol: persistence reduces
FA/h (event-wise) by 29-50% at the cost of a small event-detection-rate
drop on channels with very short (<4-sample) anomalies.

Design
------
* **Stateless function** :func:`apply_persistence` — pure numpy, O(N) via
  prefix sums, suitable for offline batch evaluation.
* **Stateful wrapper** :class:`PersistenceFilter` — keeps a per-channel
  ring of recent binary predictions so the filter can run on streaming
  blocks.  Designed to plug into :class:`CascadeDetector` as an optional
  post-L3 stage.

The filter is **off by default** (``persistence_W=0`` disables it) to
preserve backward compatibility with existing callers.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_PERSIST_W",
    "DEFAULT_PERSIST_K",
    "DEFAULT_EMA_ALPHA",
    "apply_persistence",
    "knee_threshold",
    "causal_ema",
    "PersistenceConfig",
    "PersistenceFilter",
]


DEFAULT_PERSIST_W: int = 8
DEFAULT_PERSIST_K: int = 4
# Default causal EMA alpha. Validated in experiments/diag/eval_score_smoothing.py
# on NASA-MSL/SMAP: alpha=0.2 maximises VUS-PR mean (+7.9% on MSL) without
# regressions on any channel.  alpha=1.0 disables smoothing (identity).
DEFAULT_EMA_ALPHA: float = 0.2


def apply_persistence(
    preds: np.ndarray,
    W: int = DEFAULT_PERSIST_W,
    K: int = DEFAULT_PERSIST_K,
) -> np.ndarray:
    """Causal persistence filter on a binary prediction array.

    For each sample ``i`` the output is 1 iff at least ``K`` of the last
    ``W`` samples (including ``i``) are 1.  Implemented in O(N) via prefix
    sums.

    Args:
        preds: 1-D 0/1 array of thresholded predictions.
        W: window size (number of consecutive samples considered).
        K: minimum count of positives within the window required to
            confirm an alarm.  Must satisfy ``1 <= K <= W``; otherwise
            the input is returned unchanged.

    Returns:
        1-D 0/1 int8 array of the same length as ``preds``.
    """
    if W <= 0 or K <= 0 or K > W:
        return preds.astype(np.int8, copy=False)
    preds = np.asarray(preds).astype(np.int32).ravel()
    n = len(preds)
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out
    # Prefix sums: csum[i] = sum(preds[:i]). Window sum ending at i (inclusive,
    # length W but clipped at the left boundary) = csum[i+1] - csum[max(0,i-W+1)].
    csum = np.concatenate([[0], np.cumsum(preds)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - W + 1)
    window_sums = csum[idx + 1] - csum[lo]
    out[window_sums >= K] = 1
    return out


def knee_threshold(scores: np.ndarray, eps: float = 1e-6) -> float:
    """Leak-free threshold via the Satopaa kneadle algorithm.

    Finds the "elbow" of the sorted-score curve by maximising the
    distance to the line connecting the first and last points of the
    cumulative distribution.  Works on any score distribution whose tail
    rises sharply above the bulk.

    For all-zero inputs (no non-zero scores above ``eps``), returns
    ``eps`` so any positive score counts as an alarm — the persistence
    filter is then expected to remove transient noise.

    Args:
        scores: 1-D array of scores (typically from a known-normal
            training segment).
        eps: threshold below which a score is treated as zero.

    Returns:
        A float threshold ``>= eps``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    nz = scores[scores > eps]
    if len(nz) < 10:
        return eps
    s = np.sort(nz)
    n = len(s)
    x = np.arange(n) / (n - 1)
    y = (s - s[0]) / (s[-1] - s[0] + 1e-12)
    line_y = y[0] + (y[-1] - y[0]) * x
    dist = y - line_y
    idx = int(np.argmax(dist))
    return max(float(s[idx]), eps)


def causal_ema(x: np.ndarray, alpha: float = DEFAULT_EMA_ALPHA) -> np.ndarray:
    """Causal exponential moving average smoothing.

    ``y[i] = alpha * x[i] + (1 - alpha) * y[i-1]`` with ``y[-1] = 0``.

    Applied to cascade scores *before* thresholding, this removes
    block-boundary discontinuities (each 512-sample block is scored
    independently with a pinned seed) and improves the threshold-free
    VUS-PR metric.  Validated at ``alpha=0.2`` on NASA-MSL/SMAP:
    +7.9% VUS-PR mean with no per-channel regressions.

    Args:
        x: 1-D score array.
        alpha: smoothing factor in ``(0, 1]``.  ``alpha=1.0`` returns
            the input unchanged (identity).  Smaller alpha = heavier
            smoothing but more lag.

    Returns:
        1-D float64 array of the same length.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if alpha >= 1.0:
        return x.copy()
    out = np.zeros(len(x), dtype=np.float64)
    y = 0.0
    for i, v in enumerate(x):
        y = alpha * float(v) + (1.0 - alpha) * y
        out[i] = y
    return out


class PersistenceConfig:
    """Configuration for :class:`PersistenceFilter`.

    Args:
        W: persistence window (number of consecutive samples).
        K: minimum positives within the window required to alarm.
        history_len: per-channel history buffer length.  Must be >= ``W``;
            defaults to ``W`` if smaller.
    """

    def __init__(
        self,
        W: int = DEFAULT_PERSIST_W,
        K: int = DEFAULT_PERSIST_K,
        history_len: int | None = None,
    ) -> None:
        if W < 1:
            raise ValueError(f"W must be >= 1, got {W}")
        if K < 1 or K > W:
            raise ValueError(f"K must satisfy 1 <= K <= W, got K={K} W={W}")
        self.W = int(W)
        self.K = int(K)
        self.history_len = int(max(W, history_len or 0))


class PersistenceFilter:
    """Stateful streaming persistence filter (Layer 3.5).

    Maintains a per-channel ring of recent binary predictions so the
    persistence rule can be applied across block boundaries (the online
    WarningService processes one block at a time).

    Usage::

        pf = PersistenceFilter()
        for block_preds in stream:
            confirmed = pf.update("T-4", block_preds)

    The filter is fully unsupervised — it only sees its own past binary
    decisions, never labels or future samples.

    Args:
        config: a :class:`PersistenceConfig`.  If None, a default config
                (W=8, K=4) is used.
    """

    def __init__(self, config: PersistenceConfig | None = None) -> None:
        self.config = config or PersistenceConfig()
        self._history: dict[str, np.ndarray] = {}

    def update(self, channel: str, preds: np.ndarray) -> np.ndarray:
        """Append ``preds`` to ``channel``'s history and return the
        persistence-filtered output for this block.

        The first ``W-1`` samples of the very first block may be filtered
        more aggressively than later blocks (shorter effective window);
        this is the standard causal-filter warm-up and is unavoidable
        without non-causal padding.
        """
        cfg = self.config
        preds = np.asarray(preds).astype(np.int32).ravel()
        hist = self._history.get(channel)
        if hist is None:
            combined = preds
        else:
            combined = np.concatenate([hist, preds])
        out = apply_persistence(combined, W=cfg.W, K=cfg.K)
        # Return only the tail corresponding to this block
        block_out = out[len(combined) - len(preds):]
        # Retain the last history_len samples for the next call
        keep = min(len(combined), cfg.history_len)
        self._history[channel] = combined[len(combined) - keep:].copy()
        return block_out

    def reset(self, channel: str | None = None) -> None:
        """Clear history for one channel (None = all channels)."""
        if channel is None:
            self._history.clear()
        else:
            self._history.pop(channel, None)
