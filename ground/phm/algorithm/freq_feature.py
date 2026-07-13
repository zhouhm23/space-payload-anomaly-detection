"""STFT frequency-band anomaly feature.

A complementary anomaly score derived from short-time Fourier transform
band-energy deviations.  TSPulse's reconstruction error can miss anomalies
whose time-domain shape looks normal but whose spectral content shifts
(e.g. M-5/T-4 on NASA-MSL).  This module provides the offline-baseline /
online-transform split needed for production:

* **Offline** — :meth:`FreqFeatureExtractor.fit_baseline` ingests a known
  normal training segment and returns the per-band mean/std plus the
  z-score reference range (``z_min``/``z_max``) that characterise the
  healthy spectrum.
* **Online** — :meth:`FreqFeatureExtractor.transform` scores a new test
  segment against the stored baseline: z-score of band energy, max-pooled
  across bands, then **linearly mapped onto ``[0, 1]`` using the stored
  ``z_min``/``z_max`` and clipped**.

The ``z_min``/``z_max`` reference is the critical bit.  Earlier versions
fit a ``MinMaxScaler`` on whatever array ``transform`` received, which
produced different scales for the offline full-segment call vs the online
512-point streaming block — calibration thresholds chosen offline became
unreachable online (see ``experiments/calibration/diag_calibration_scale.py``
for the reproduction).  Using a fixed reference range derived from the
training segment makes the offline and online scales mathematically
identical.

The baseline (``band_mean``/``band_std``/``z_min``/``z_max``) must be
supplied at construction time (from ``channel_calibration.json``).
Calling :meth:`transform` without a baseline raises ``ValueError`` —
this is intentional: a frequency score computed without a normal
reference is meaningless, and we want to fail loud rather than silently
degrade.

Algorithm ported verbatim (logic + constants) from
``experiments/tspulse_eval/freq_feature_full.py:73-105`` where it was
validated on NASA-MSL (event detection 0.76→0.963 when combined with
per-channel selection).  The MinMax→fixed-reference normalisation is a
later fix to a scale-drift bug, not part of the original experiment.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

__all__ = ["FreqFeatureExtractor"]


def _stft_band_power(ts: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """STFT → per-band × per-window power matrix ``|Zxx|²``.

    ``boundary=None`` disables scipy's default zero-padding at the signal
    ends.  With padding, each streaming block's first window is zero-filled
    and its spectrum diverges from the same window computed over the full
    segment — exactly the offline/online mismatch we are trying to fix.
    No padding → the i-th STFT window depends only on samples
    ``[i*hop, i*hop+nperseg)``, identical whether the call sees the whole
    segment or just one block.
    """
    _, _, Zxx = stft(
        ts, fs=1.0, nperseg=nperseg, noverlap=noverlap, boundary=None
    )
    return np.abs(Zxx) ** 2  # (n_freqs, n_windows)


def _point_zscores(
    ts: np.ndarray,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    """Band z-score, max-pooled across bands, broadcast to point level.

    Shared kernel so :func:`fit_baseline` and :meth:`FreqFeatureExtractor.transform`
    compute the z-score the same way — the only difference is which array
    the baseline/reference is drawn from.
    """
    n = len(ts)
    power = _stft_band_power(ts, nperseg, noverlap)
    z = np.abs(power - band_mean[:, None]) / band_std[:, None]
    window_scores = z.max(axis=0)  # (n_windows,)

    point_scores = np.zeros(n, dtype=np.float64)
    hop = nperseg - noverlap
    for i, ws in enumerate(window_scores):
        start = i * hop
        end = min(start + nperseg, n)
        if start >= n:
            break
        point_scores[start:end] = np.maximum(point_scores[start:end], ws)
    return point_scores


class FreqFeatureExtractor:
    """STFT frequency-band anomaly scorer with offline baseline.

    Args:
        nperseg: STFT segment length (default 64, matches the validated
            experiment configuration).
        noverlap: STFT overlap (default 32 → 50% overlap, hop=32).
        band_mean: per-frequency-band mean power from the normal baseline.
            Required for :meth:`transform`.  Pass ``None`` only when using
            this instance solely for :meth:`fit_baseline`.
        band_std: per-frequency-band std power from the normal baseline.
            Same requirement as ``band_mean``.
        z_min / z_max: reference range for the band z-score, derived from
            the training segment by :meth:`fit_baseline`.  ``transform``
            maps ``(z - z_min) / (z_max - z_min)`` and clips to ``[0, 1]``.
            When ``None``, :meth:`transform` falls back to a per-call
            ``MinMaxScaler`` (legacy behaviour) and emits a warning — this
            is kept only for backward compatibility with old calibration
            JSONs that lack the ``freq_z_min``/``freq_z_max`` fields.
    """

    def __init__(
        self,
        nperseg: int = 64,
        noverlap: int = 32,
        band_mean: Any | None = None,
        band_std: Any | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> None:
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.band_mean = (
            np.asarray(band_mean, dtype=np.float64) if band_mean is not None else None
        )
        self.band_std = (
            np.asarray(band_std, dtype=np.float64) if band_std is not None else None
        )
        self.z_min = float(z_min) if z_min is not None else None
        self.z_max = float(z_max) if z_max is not None else None

    # ------------------------------------------------------------------
    # Offline baseline fitting
    # ------------------------------------------------------------------

    @staticmethod
    def fit_baseline(
        train_ts: np.ndarray,
        nperseg: int = 64,
        noverlap: int = 32,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Compute per-frequency-band mean/std + z-score range from a normal segment.

        Returns ``(band_mean, band_std, z_min, z_max)``:

        * ``band_mean``/``band_std`` — 1-D arrays of length ``nperseg // 2 + 1``
          (the number of STFT frequency bins), serialised into
          ``channel_calibration.json``.
        * ``z_min``/``z_max`` — scalar range of the band z-score over the
          training segment.  :meth:`transform` uses these to map new scores
          onto ``[0, 1]`` so the offline and online scales match.
        """
        train_ts = np.asarray(train_ts, dtype=np.float64).ravel()
        train_power = _stft_band_power(train_ts, nperseg, noverlap)
        band_mean = train_power.mean(axis=1)
        band_std = train_power.std(axis=1) + 1e-8

        # Reference z-score range over the training segment — the online
        # path will use exactly these values, so the scales are identical.
        train_z = _point_zscores(train_ts, band_mean, band_std, nperseg, noverlap)
        z_min = float(np.min(train_z))
        z_max = float(np.max(train_z))
        if z_max - z_min < 1e-8:
            # Degenerate (near-constant) train segment — widen to avoid div-by-zero.
            z_max = z_min + 1.0
        return band_mean, band_std, z_min, z_max

    # ------------------------------------------------------------------
    # Online scoring
    # ------------------------------------------------------------------

    def transform(self, test_ts: np.ndarray) -> np.ndarray:
        """Score a test segment against the stored baseline.

        Args:
            test_ts: 1-D telemetry array.

        Returns:
            1-D float32 array of length ``len(test_ts)`` in ``[0, 1]``
            (higher = more spectrally anomalous).  Scores are mapped from
            the band z-score using the stored ``z_min``/``z_max`` reference
            and clipped — this makes the offline (full-segment) and online
            (per-block) scales identical.

        Raises:
            ValueError: if no baseline was supplied at construction.
        """
        if self.band_mean is None or self.band_std is None:
            raise ValueError(
                "FreqFeatureExtractor.transform requires a baseline — "
                "pass band_mean/band_std from fit_baseline() or "
                "channel_calibration.json."
            )
        test_ts = np.asarray(test_ts, dtype=np.float64).ravel()
        n_test = len(test_ts)
        if n_test == 0:
            return np.zeros(0, dtype=np.float32)

        point_scores = _point_zscores(
            test_ts, self.band_mean, self.band_std, self.nperseg, self.noverlap
        )

        if self.z_min is not None and self.z_max is not None:
            # Fixed-reference normalisation — offline/online identical.
            denom = self.z_max - self.z_min
            if denom > 1e-8:
                point_scores = (point_scores - self.z_min) / denom
            else:
                point_scores = np.zeros_like(point_scores)
            point_scores = np.clip(point_scores, 0.0, 1.0)
        else:
            # Legacy fallback: per-call MinMax.  Emits a warning because
            # thresholds calibrated against this are scale-unstable.
            logger.warning(
                "FreqFeatureExtractor.transform called without z_min/z_max — "
                "falling back to per-call MinMax. Recalibrate with "
                "fit_baseline to embed the reference range."
            )
            point_scores = (
                MinMaxScaler().fit_transform(point_scores.reshape(-1, 1)).ravel()
            )
        return point_scores.astype(np.float32)
