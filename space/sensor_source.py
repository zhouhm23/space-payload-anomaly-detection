"""Simulated sensor data source — mimics a real data acquisition card.

Two modes:
  1. ``DatasetSource``  — replays NASA-SMAP/MSL telemetry; outputs 0 after
     the dataset is exhausted (simulating sensor end-of-life / shutdown).
  2. ``SyntheticSource`` — generates continuous synthetic signals (sine,
     square, multi-harmonic) that never run out.  Useful for stress-testing
     the processing pipeline without depending on fixed-length data.

Both sources expose the same ``read()`` interface, so the space-segment code
does not need to know which source is active — just like a real DAQ card.

Noise injection (missing values, Gaussian noise, sampling jitter, clipping)
is applied at the source level, simulating real sensor artefacts *before*
the signal reaches the preprocessing pipeline.
"""

from __future__ import annotations

import os
import sys
import time
import math
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Resolve project root for data_loader import
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import list_channels, load_channel, load_train


@dataclass
class SensorNoiseConfig:
    """Sensor artefact parameters applied at the acquisition layer.

    All values default to 0 (clean signal) so the source can be used
    without noise for baseline testing.
    """
    missing_rate: float = 0.0       # fraction of samples set to NaN
    noise_std: float = 0.0          # additive Gaussian noise std
    jitter_std: float = 0.0         # sampling-time jitter (in samples)
    clip_range: tuple[float, float] | None = None  # sensor saturation
    dropout_gap_mean: int = 0       # mean contiguous gap length (0=isolated)
    random_seed: int | None = None


def _apply_noise(values: np.ndarray, cfg: SensorNoiseConfig) -> np.ndarray:
    """Inject sensor artefacts into a chunk of values."""
    rng = np.random.default_rng(cfg.random_seed)
    raw = values.astype(np.float64).copy()

    # Advance RNG state deterministically per call for reproducibility
    if cfg.random_seed is not None:
        cfg.random_seed += 1

    if cfg.noise_std > 0:
        raw += rng.normal(0, cfg.noise_std, size=len(raw))

    if cfg.clip_range is not None:
        raw = np.clip(raw, cfg.clip_range[0], cfg.clip_range[1])

    if cfg.missing_rate > 0:
        mask = np.zeros(len(raw), dtype=bool)
        if cfg.dropout_gap_mean > 0:
            n_gaps = int(len(raw) * cfg.missing_rate / max(cfg.dropout_gap_mean, 1))
            for _ in range(n_gaps):
                start = int(rng.integers(0, len(raw)))
                gap_len = max(1, int(rng.exponential(cfg.dropout_gap_mean)))
                mask[start : min(start + gap_len, len(raw))] = True
        else:
            mask = rng.random(len(raw)) < cfg.missing_rate
        raw[mask] = np.nan

    if cfg.jitter_std > 0:
        n = len(raw)
        uniform_t = np.arange(n, dtype=float)
        jittered_t = uniform_t + rng.normal(0, cfg.jitter_std, size=n)
        valid = ~np.isnan(raw)
        if valid.sum() > 2:
            raw = np.interp(uniform_t, jittered_t[valid], raw[valid],
                            left=raw[valid][0], right=raw[valid][-1])

    return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SensorSource(ABC):
    """Abstract sensor data source — mimics a DAQ card's ``read()`` interface."""

    @abstractmethod
    def read(self, n: int) -> np.ndarray:
        """Read ``n`` samples. Returns float32 array of length ``n``.

        May contain NaN for missing values (if noise injection is enabled).
        Returns all-zeros when the source is exhausted (dataset mode).
        """
        ...

    @property
    @abstractmethod
    def exhausted(self) -> bool:
        """True when no more real data is available (dataset mode only)."""
        ...

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Identifier of the current channel being streamed."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Nominal sample rate in Hz."""
        ...


# ---------------------------------------------------------------------------
# Dataset replay source
# ---------------------------------------------------------------------------

class DatasetSource(SensorSource):
    """Replays a NASA-SMAP/MSL telemetry channel as a live sensor stream.

    - Normal mode (``sample_rate > 0``): reads ``n`` samples per call,
      respecting the configured rate.  Pacing is done by the caller.
    - Bulk mode (``sample_rate == -1``): the first ``read()`` returns
      **all remaining data** regardless of ``n``, then marks exhausted
      (all subsequent reads return zeros).

    After the dataset is exhausted, ``read()`` returns zeros — simulating
    a sensor that has gone offline or been shut down.
    """

    def __init__(
        self,
        dataset: str = "NASA-MSL",
        channel: str | None = None,
        sample_rate: float = 1.0,
        noise: SensorNoiseConfig | None = None,
    ):
        channels = list_channels(dataset)
        if not channels:
            raise ValueError(f"No channels found in dataset {dataset}")

        if channel is not None:
            match = [c for c in channels if c[0] == channel]
            if not match:
                raise ValueError(f"Channel {channel} not in {dataset}")
            ch_name, train_path, test_path = match[0]
        else:
            ch_name, train_path, test_path = channels[0]

        self._channel = ch_name
        self._dataset = dataset
        self._sample_rate = sample_rate
        self._noise = noise or SensorNoiseConfig()

        test_ts, test_labels = load_channel(test_path, train_path)
        self._data = test_ts.astype(np.float32)
        self._labels = test_labels
        self._pos = 0
        self._exhausted = False

    def read(self, n: int) -> np.ndarray:
        if self._exhausted:
            return np.empty(0, dtype=np.float32)

        # bulk mode — return everything remaining, then done
        if self._sample_rate < 0:
            remaining = self._data[self._pos:].copy()
            self._pos = len(self._data)
            self._exhausted = True
            if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
                remaining = _apply_noise(remaining, self._noise)
            return remaining

        # normal mode — return actual available, may be shorter than n
        available = min(n, len(self._data) - self._pos)
        if available == 0:
            self._exhausted = True
            return np.empty(0, dtype=np.float32)
        chunk = self._data[self._pos : self._pos + available].copy()
        self._pos += available
        if self._pos >= len(self._data):
            self._exhausted = True

        if self._pos >= len(self._data):
            self._exhausted = True

        if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
            chunk = _apply_noise(chunk, self._noise)

        return chunk

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def channel_name(self) -> str:
        return self._channel

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def labels(self) -> np.ndarray:
        """Ground-truth labels for the replayed portion (testing only)."""
        return self._labels[: self._pos]

    def reset(self):
        """Rewind to the beginning of the dataset."""
        self._pos = 0
        self._exhausted = False


# ---------------------------------------------------------------------------
# Synthetic signal source
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    """Parameters for synthetic signal generation."""
    signal_type: str = "multi_sine"   # "sine", "square", "multi_sine", "chirp"
    frequency: float = 0.02           # cycles per sample (primary)
    amplitude: float = 1.0
    offset: float = 0.0
    noise_floor: float = 0.0          # baseline noise added to clean signal
    anomaly_every: int = 0            # inject a spike every N samples (0=off)
    anomaly_magnitude: float = 3.0


class SyntheticSource(SensorSource):
    """Generates continuous synthetic sensor signals.

    Unlike ``DatasetSource``, this source never exhausts — it will keep
    producing samples indefinitely, making it ideal for long-running
    pipeline tests.
    """

    def __init__(
        self,
        config: SyntheticConfig | None = None,
        sample_rate: float = 1.0,
        noise: SensorNoiseConfig | None = None,
    ):
        self._config = config or SyntheticConfig()
        self._sample_rate = sample_rate
        self._noise = noise or SensorNoiseConfig()
        self._t = 0  # global sample counter
        self._channel = f"SYN-{self._config.signal_type}"

    def read(self, n: int) -> np.ndarray:
        cfg = self._config
        t = np.arange(self._t, self._t + n, dtype=np.float64)

        if cfg.signal_type == "sine":
            signal = cfg.amplitude * np.sin(2 * np.pi * cfg.frequency * t) + cfg.offset
        elif cfg.signal_type == "square":
            signal = cfg.amplitude * np.sign(np.sin(2 * np.pi * cfg.frequency * t)) + cfg.offset
        elif cfg.signal_type == "multi_sine":
            f1, f2, f3 = cfg.frequency, cfg.frequency * 2.3, cfg.frequency * 5.7
            signal = (cfg.amplitude * 0.5 * np.sin(2 * np.pi * f1 * t)
                      + cfg.amplitude * 0.3 * np.sin(2 * np.pi * f2 * t)
                      + cfg.amplitude * 0.2 * np.sin(2 * np.pi * f3 * t))
            signal += cfg.offset
        elif cfg.signal_type == "chirp":
            k = cfg.frequency / max(n, 1)
            phase = 2 * np.pi * (cfg.frequency * t + 0.5 * k * t * t)
            signal = cfg.amplitude * np.sin(phase) + cfg.offset
        else:
            signal = np.full(n, cfg.offset, dtype=np.float64)

        if cfg.noise_floor > 0:
            signal += np.random.default_rng().normal(0, cfg.noise_floor, n)

        if cfg.anomaly_every > 0:
            spike_mask = (t.astype(int) % cfg.anomaly_every) < 3
            signal[spike_mask] += cfg.anomaly_magnitude

        self._t += n

        signal = signal.astype(np.float32)
        if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
            signal = _apply_noise(signal, self._noise)

        return signal

    @property
    def exhausted(self) -> bool:
        return False  # synthetic source never exhausts

    @property
    def channel_name(self) -> str:
        return self._channel

    @property
    def sample_rate(self) -> float:
        return self._sample_rate
