"""Joint (cross-channel) anomaly detection via co-anomaly consensus.

The device-tree folder hierarchy defines subsystem grouping: channels in the
same folder are physically/logically related (e.g. sensors on the same
subsystem).  When multiple sibling channels simultaneously exceed their
respective anomaly thresholds, that co-occurrence is stronger evidence of a
real subsystem-level anomaly than any single channel's score alone.

This module implements **decision-level fusion** (co-anomaly consensus) rather
than feature-level multivariate detection.  The per-channel univariate cascade
(TSPulse + L1 + L3) runs unchanged; this layer consumes the per-channel
final scores and emits a folder-level joint score.  This avoids retraining
models and leverages the existing per-channel calibration (flip / score-type /
threshold).

The consensus metric at each time point ``t`` is::

    joint[t] = mean( score_ch[t] > threshold_ch  for ch in channels )

Returns a float in ``[0, 1]``: 0 = no channel exceeds its threshold, 1 = all
channels exceed.  A joint score above ``joint_threshold`` (default 0.5, i.e.
majority of channels agree) triggers a subsystem-level alert.
"""

from __future__ import annotations

import numpy as np


def co_anomaly_consensus(
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> np.ndarray:
    """Compute the co-anomaly consensus score across channels.

    Args:
        scores: ``{channel_name: 1-D anomaly score array}``.  All channels
            in the same folder.  Arrays may have different lengths — they
            are truncated to the shortest (keeping the most-recent tail)
            so array indices are temporally aligned.
        thresholds: ``{channel_name: per-channel threshold}``.  Each
            channel uses its own calibrated threshold for fair comparison
            (different channels have different baseline noise levels).
            Missing entries default to 0.5.

    Returns:
        1-D ``np.ndarray`` of floats in ``[0, 1]``, length = shortest input.
        Returns an empty array if fewer than 2 channels are provided (joint
        detection requires at least 2 siblings to be meaningful).
    """
    channels = list(scores.keys())
    if len(channels) < 2:
        return np.zeros(0, dtype=np.float32)

    # Align lengths: truncate to the shortest channel's array.
    min_len = min(len(scores[ch]) for ch in channels)
    if min_len == 0:
        return np.zeros(0, dtype=np.float32)

    flags = np.zeros(min_len, dtype=np.float32)
    for ch in channels:
        s = np.asarray(scores[ch][:min_len], dtype=np.float32)
        thr = thresholds.get(ch, 0.5)
        flags += (s > thr).astype(np.float32)

    return flags / len(channels)


__all__ = ["co_anomaly_consensus"]
