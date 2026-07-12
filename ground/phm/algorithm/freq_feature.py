"""STFT frequency-band anomaly feature.

A complementary anomaly score derived from short-time Fourier transform
band-energy deviations.  TSPulse's reconstruction error can miss anomalies
whose time-domain shape looks normal but whose spectral content shifts
(e.g. M-5/T-4 on NASA-MSL).  This module provides the offline-baseline /
online-transform split needed for production:

* **Offline** — :meth:`FreqFeatureExtractor.fit_baseline` ingests a known
  normal training segment and returns the per-band mean/std that characterise
  the healthy spectrum.
* **Online** — :meth:`FreqFeatureExtractor.transform` scores a new test
  segment against the stored baseline (z-score of band energy, max-pooled
  across bands, then MinMax-normalised to ``[0,1]``).

The baseline must be supplied at construction time (from
``channel_calibration.json``).  Calling :meth:`transform` without a baseline
raises ``ValueError`` — this is intentional: a frequency score computed
without a normal reference is meaningless, and we want to fail loud rather
than silently degrade.

Algorithm ported verbatim (logic + constants) from
``experiments/tspulse_eval/freq_feature_full.py:73-105`` where it was
validated on NASA-MSL (event detection 0.76→0.963 when combined with
per-channel selection).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler

__all__ = ["FreqFeatureExtractor"]


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
    """

    def __init__(
        self,
        nperseg: int = 64,
        noverlap: int = 32,
        band_mean: Any | None = None,
        band_std: Any | None = None,
    ) -> None:
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.band_mean = (
            np.asarray(band_mean, dtype=np.float64) if band_mean is not None else None
        )
        self.band_std = (
            np.asarray(band_std, dtype=np.float64) if band_std is not None else None
        )

    # ------------------------------------------------------------------
    # Offline baseline fitting
    # ------------------------------------------------------------------

    @staticmethod
    def fit_baseline(
        train_ts: np.ndarray,
        nperseg: int = 64,
        noverlap: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-frequency-band mean/std from a normal training segment.

        Returns ``(band_mean, band_std)`` — two 1-D arrays of length
        ``nperseg // 2 + 1`` (the number of STFT frequency bins).  These
        should be serialised into ``channel_calibration.json`` and passed
        back to the constructor at runtime.
        """
        train_ts = np.asarray(train_ts, dtype=np.float64).ravel()
        _, _, Zxx = stft(train_ts, fs=1.0, nperseg=nperseg, noverlap=noverlap)
        train_power = np.abs(Zxx) ** 2  # (n_freqs, n_windows)
        band_mean = train_power.mean(axis=1)
        band_std = train_power.std(axis=1) + 1e-8
        return band_mean, band_std

    # ------------------------------------------------------------------
    # Online scoring
    # ------------------------------------------------------------------

    def transform(self, test_ts: np.ndarray) -> np.ndarray:
        """Score a test segment against the stored baseline.

        Args:
            test_ts: 1-D telemetry array.

        Returns:
            1-D float32 array of length ``len(test_ts)``, MinMax-normalised
            to ``[0, 1]`` (higher = more spectrally anomalous).

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

        _, _, Zxx_te = stft(
            test_ts, fs=1.0, nperseg=self.nperseg, noverlap=self.noverlap
        )
        test_power = np.abs(Zxx_te) ** 2  # (n_freqs, n_windows)

        # z-score of each band, then max-pool across bands per window
        z = np.abs(test_power - self.band_mean[:, None]) / self.band_std[:, None]
        window_scores = z.max(axis=0)  # (n_windows,)

        # Broadcast window scores back to point level via max-pooling
        hop = self.nperseg - self.noverlap
        point_scores = np.zeros(n_test, dtype=np.float32)
        for i, ws in enumerate(window_scores):
            start = i * hop
            end = min(start + self.nperseg, n_test)
            if start >= n_test:
                break
            point_scores[start:end] = np.maximum(point_scores[start:end], ws)

        # MinMax normalise to [0, 1] — matches the experiment pipeline so
        # the score is comparable to TSPulse's MinMax-normalised output.
        point_scores = (
            MinMaxScaler().fit_transform(point_scores.reshape(-1, 1)).ravel()
        )
        return point_scores.astype(np.float32)
