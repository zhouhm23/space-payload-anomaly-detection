"""Unified sensor data source — mimics a real data acquisition card.

All data sources expose the same ``read()`` interface, so the space-segment
code does not need to know whether the data comes from a local file or a
virtual (synthetic) sensor — just like a real DAQ card.

Sources are registered in a global registry and can be looked up by ID:
  - ``file:NASA-MSL/C-1``  — replay a NASA telemetry channel
  - ``virtual:sine``        — continuous sine wave generator
  - ``virtual:multi_sine``  — multi-harmonic signal generator
  - ``virtual:square``      — square wave generator
  - ``virtual:chirp``       — chirp signal generator

Noise injection (missing values, Gaussian noise, sampling jitter, clipping)
is applied at the source level, simulating real sensor artefacts *before*
the signal reaches the preprocessing pipeline.
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Sequence

# Resolve project root for data_loader import
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import list_channels, load_channel, load_train

# ---------------------------------------------------------------------------
# Source ID conventions
# ---------------------------------------------------------------------------
FILE_PREFIX = "file:"
VIRTUAL_PREFIX = "virtual:"


def _make_file_id(dataset: str, channel: str) -> str:
    return f"{FILE_PREFIX}{dataset}/{channel}"


def _make_virtual_id(signal: str) -> str:
    return f"{VIRTUAL_PREFIX}{signal}"


def _parse_file_id(source_id: str) -> tuple[str, str]:
    """Parse ``file:NASA-MSL/C-1`` → (dataset, channel)."""
    body = source_id[len(FILE_PREFIX):]
    parts = body.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid file source ID: {source_id}")
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Noise configuration (unchanged)
# ---------------------------------------------------------------------------

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
        Returns empty array when the source is exhausted.
        """
        ...

    @property
    @abstractmethod
    def exhausted(self) -> bool:
        """True when no more real data is available."""
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

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique source identifier (e.g. ``file:NASA-MSL/C-1``)."""
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable display label for UI dropdowns."""
        ...

    @property
    @abstractmethod
    def is_virtual(self) -> bool:
        """True if this is a virtual (synthetic) sensor, False for file replay."""
        ...


# ---------------------------------------------------------------------------
# Source registry (extensible factory — replaces if/elif chains)
# ---------------------------------------------------------------------------
# Each concrete source registers itself under a prefix via the
# ``@register_source`` decorator. ``create_source`` then becomes a single dict
# lookup, so adding a new source type only requires decorating the new class —
# no edits to the factory.
#
# A registered class must expose a ``from_source_id(source_id, **kwargs)``
# classmethod that parses the source_id and returns an instance.  This moves
# the parsing logic next to the class that owns it.

_SOURCE_REGISTRY: dict[str, type[SensorSource]] = {}


def register_source(prefix: str):
    """Class decorator: register a ``SensorSource`` subclass under ``prefix``.

    The class must implement ``from_source_id(cls, source_id, **kwargs)``.
    """
    def decorator(cls: type[SensorSource]) -> type[SensorSource]:
        if prefix in _SOURCE_REGISTRY:
            raise RuntimeError(
                f"Source prefix {prefix!r} already registered "
                f"to {_SOURCE_REGISTRY[prefix].__name__}"
            )
        _SOURCE_REGISTRY[prefix] = cls
        return cls
    return decorator


def registered_source_prefixes() -> list[str]:
    """Return all registered source-id prefixes (for diagnostics)."""
    return sorted(_SOURCE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# File replay source (was DatasetSource)
# ---------------------------------------------------------------------------

@register_source(FILE_PREFIX)
class FileSource(SensorSource):
    """Replays a NASA-SMAP/MSL telemetry channel as a live sensor stream.

    - Normal mode (``sample_rate > 0``): reads ``n`` samples per call,
      respecting the configured rate.  Pacing is done by the caller.
    - Bulk mode (``sample_rate == -1``): the first ``read()`` returns
      **all remaining data** regardless of ``n``, then marks exhausted.
    - Loop mode (``loop=True``): when the dataset is exhausted, it
      automatically rewinds to the beginning (debugging convenience).

    After the dataset is exhausted and loop is off, ``read()`` returns
    an empty array — simulating a sensor that has gone offline.
    """

    def __init__(
        self,
        dataset: str = "NASA-MSL",
        channel: str | None = None,
        sample_rate: float = 1.0,
        noise: SensorNoiseConfig | None = None,
        loop: bool = False,
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
        self._source_id = _make_file_id(dataset, ch_name)
        self._loop = loop

        test_ts, test_labels = load_channel(test_path, train_path)
        self._data = test_ts.astype(np.float32)
        self._labels = test_labels
        self._pos = 0
        self._exhausted = False

    def read(self, n: int) -> np.ndarray:
        if self._exhausted and not self._loop:
            return np.empty(0, dtype=np.float32)

        # bulk mode — return everything remaining, then done
        if self._sample_rate < 0:
            remaining = self._data[self._pos:].copy()
            self._pos = len(self._data)
            self._exhausted = True
            if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
                remaining = _apply_noise(remaining, self._noise)
            return remaining

        data_len = len(self._data)

        # non-loop: 读完即止
        if not self._loop:
            available = min(n, data_len - self._pos)
            if available == 0:
                self._exhausted = True
                return np.empty(0, dtype=np.float32)
            chunk = self._data[self._pos : self._pos + available].copy()
            self._pos += available
            if self._pos >= data_len:
                self._exhausted = True
            if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
                chunk = _apply_noise(chunk, self._noise)
            return chunk

        # loop 模式：块式读取，每次固定返回 n 点，不足时从文件头补齐
        result = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            chunk_size = min(n - filled, data_len - self._pos)
            result[filled : filled + chunk_size] = self._data[self._pos : self._pos + chunk_size]
            filled += chunk_size
            self._pos += chunk_size
            if self._pos >= data_len:
                self._pos = 0  # 回到文件头继续读

        if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
            result = _apply_noise(result, self._noise)
        return result

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
    def source_id(self) -> str:
        return self._source_id

    @property
    def label(self) -> str:
        return f"📂 {self._dataset} · {self._channel}"

    @property
    def is_virtual(self) -> bool:
        return False

    @property
    def labels(self) -> np.ndarray:
        """Ground-truth labels for the replayed portion (testing only)."""
        return self._labels[: self._pos]

    def reset(self):
        """Rewind to the beginning of the dataset."""
        self._pos = 0
        self._exhausted = False

    @classmethod
    def from_source_id(
        cls,
        source_id: str,
        *,
        sample_rate: float = 1.0,
        noise: SensorNoiseConfig | None = None,
        loop: bool = False,
        **_unused,
    ) -> "FileSource":
        """Parse ``file:NASA-MSL/C-1`` and build a FileSource."""
        dataset, channel = _parse_file_id(source_id)
        return cls(
            dataset=dataset,
            channel=channel,
            sample_rate=sample_rate,
            noise=noise,
            loop=loop,
        )


# ---------------------------------------------------------------------------
# Synthetic signal configuration (unchanged)
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
    # Random anomaly injection: each read() call has ``anomaly_prob`` chance
    # of injecting a realistic degradation pattern into the block.
    # Types:
    # - "drift": gradual baseline offset (thermal drift, sensor aging)
    # - "spike": short burst of high-amplitude noise (switching transient)
    # - "zero_drop": random values become 0 (sensor open-circuit / signal loss)
    # This makes the synthetic source produce a MIX of normal and anomalous
    # data — essential for testing false-alarm filtering, because a source
    # that is always normal only generates false positives (TSPulse misfires)
    # but never true positives.
    anomaly_prob: float = 0.0         # 0=off, 0.15≈every 6-7 blocks
    anomaly_type: str = "drift"       # "drift" | "spike" | "zero_drop"
    random_seed: int | None = None


# ---------------------------------------------------------------------------
# Virtual sensor source (was SyntheticSource)
# ---------------------------------------------------------------------------

# Preset virtual sensors registered in the global list
# 异常注入策略：用 zero_drop（随机值变 0，模拟传感器失联），不会超量程
# signal 在 return 前会 clip 到 [-amplitude, amplitude]，保证不超量程
_VIRTUAL_PRESETS: dict[str, SyntheticConfig] = {
    "sine": SyntheticConfig(
        signal_type="sine", frequency=0.02, amplitude=1.0,
        anomaly_prob=0.12, anomaly_type="zero_drop", anomaly_magnitude=1.0,
        random_seed=42,
    ),
    "square": SyntheticConfig(signal_type="square", frequency=0.02, amplitude=1.0),
    "multi_sine": SyntheticConfig(
        signal_type="multi_sine", frequency=0.02, amplitude=1.0,
        anomaly_prob=0.15, anomaly_type="zero_drop", anomaly_magnitude=1.0,
        random_seed=99,
    ),
    "chirp": SyntheticConfig(signal_type="chirp", frequency=0.01, amplitude=1.0),
}

_VIRTUAL_LABELS = {
    "sine": "🎛️ 虚拟传感器 · 正弦波",
    "square": "🎛️ 虚拟传感器 · 方波",
    "multi_sine": "🎛️ 虚拟传感器 · 多谐波",
    "chirp": "🎛️ 虚拟传感器 · 啁啾",
}


@register_source(VIRTUAL_PREFIX)
class VirtualSensorSource(SensorSource):
    """Generates continuous synthetic sensor signals.

    Unlike ``FileSource``, this source never exhausts — it will keep
    producing samples indefinitely, making it ideal for long-running
    pipeline tests and demo scenarios where no real sensor is available.

    The DAQ card (ie the space-segment code) does not know this sensor is
    virtual — it just calls ``read()`` and gets data.
    """

    def __init__(
        self,
        signal_type: str = "multi_sine",
        config: SyntheticConfig | None = None,
        sample_rate: float = 1.0,
        noise: SensorNoiseConfig | None = None,
        signal_freq_hz: float | None = None,
    ):
        self._config = config or _VIRTUAL_PRESETS.get(
            signal_type, SyntheticConfig(signal_type=signal_type),
        )
        # signal_freq_hz (Hz) → frequency (cycles per sample)
        if signal_freq_hz is not None and sample_rate > 0:
            self._config.frequency = signal_freq_hz / sample_rate
        self._signal_type = signal_type
        self._sample_rate = sample_rate
        self._noise = noise or SensorNoiseConfig()
        self._t = 0  # global sample counter
        self._channel = f"VS-{self._config.signal_type}"
        self._source_id = _make_virtual_id(signal_type)

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

        # Random anomaly injection — makes the synthetic source produce a
        # realistic mix of normal and anomalous blocks.  Without this, the
        # source is always normal and only tests TSPulse false positives,
        # never true positives.
        if cfg.anomaly_prob > 0:
            rng = np.random.default_rng(cfg.random_seed)
            cfg.random_seed = (cfg.random_seed or 0) + 1
            if rng.random() < cfg.anomaly_prob:
                if cfg.anomaly_type == "drift":
                    # Gradual baseline drift: signal offset ramps up over
                    # the block, simulating slow degradation (e.g. thermal
                    # drift, sensor aging).  Harder to detect than a spike.
                    ramp = np.linspace(0, cfg.anomaly_magnitude, n)
                    signal += ramp
                elif cfg.anomaly_type == "zero_drop":
                    # Zero-drop: random points in the block become 0,
                    # simulating sensor open-circuit / signal loss / ADC dropout.
                    # ~15-25% of the block affected, scattered (not contiguous).
                    drop_rate = 0.15 + rng.random() * 0.10  # 15%~25%
                    drop_mask = rng.random(n) < drop_rate
                    signal[drop_mask] = 0.0
                else:
                    # Spike burst (default): a short segment of high-amplitude
                    # noise, simulating a sudden transient (e.g. switching event,
                    # particle hit).  ~15% of the block affected.
                    burst_start = int(rng.integers(0, max(1, n // 2)))
                    burst_len = max(1, n // 8)
                    burst = rng.normal(0, cfg.anomaly_magnitude, burst_len)
                    end = min(burst_start + burst_len, n)
                    signal[burst_start:end] += burst[:end - burst_start]

        # 超量程截断：所有异常注入后，clip 到 [-amplitude, +amplitude]
        # 模拟真实 ADC 饱和特性，避免 drift/spike 让信号超出传感器量程
        if cfg.amplitude > 0:
            signal = np.clip(signal, -cfg.amplitude, cfg.amplitude)

        self._t += n

        signal = signal.astype(np.float32)
        if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
            signal = _apply_noise(signal, self._noise)

        return signal

    @property
    def exhausted(self) -> bool:
        return False  # virtual sensor never exhausts

    @property
    def channel_name(self) -> str:
        return self._channel

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def label(self) -> str:
        return _VIRTUAL_LABELS.get(self._signal_type, f"🎛️ 虚拟传感器 · {self._signal_type}")

    @property
    def is_virtual(self) -> bool:
        return True

    @classmethod
    def from_source_id(
        cls,
        source_id: str,
        *,
        sample_rate: float = 1.0,
        noise: SensorNoiseConfig | None = None,
        signal_freq_hz: float | None = None,
        **_unused,
    ) -> "VirtualSensorSource":
        """Parse ``virtual:sine`` and build a VirtualSensorSource."""
        signal_type = source_id[len(VIRTUAL_PREFIX):]
        if signal_type not in _VIRTUAL_PRESETS:
            raise ValueError(
                f"Unknown virtual sensor: {signal_type}. "
                f"Available: {list(_VIRTUAL_PRESETS.keys())}"
            )
        return cls(
            signal_type=signal_type,
            sample_rate=sample_rate,
            noise=noise,
            signal_freq_hz=signal_freq_hz,
        )


# ---------------------------------------------------------------------------
# C-MAPSS engine degradation source
# ---------------------------------------------------------------------------
# 需求：CMAPSS-1（RUL 退化演示）也走信号发生器 → IPC → 采集卡 → TCP → Django，
# 与 C-1/D-14 同链路。本类把 C-MAPSS test_FD00X.txt 中某台 engine 的退化曲线
# 按 cycle 顺序展开为 1D float 流（每 cycle 一个点），让 ring_buffer/SQLite/异常
# 检测全套链路都看得见它。RUL 精确预测仍由 RulService 直接读全 14 维做（研究
# 用途，对真实场景也合理——RUL 模型需要完整 14 维特征）。
#
# 派生 1D 信号的选择：用 14 维传感器的 z-score 归一化均值（按整个 engine 序列
# 计算均值/方差）。这个标量随 cycle 单调变化，能直观反映「退化趋势」。

CMAPSS_PREFIX = "cmapss:"

# C-MAPSS 数据集根目录（与 RulService 同源）。
# 路径推演：__file__ = 生产实习/src/space/sensor_source.py
#   parent                  = 生产实习/src/space
#   parent.parent           = 生产实习/src
#   parent.parent.parent    = 生产实习/  ← datasets/ 同级
_CMAPSS_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "CMAPSSData"

# 14 个携带退化信息的传感器列（与 RulService.CMAPSSDataSource.SENSORS 一致）
_CMAPSS_DEGRADATION_SENSORS = (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)

# C-MAPSS test_FD00X.txt 的 26 列（unit + cycle + 3 op + 21 sensors）
_CMAPSS_COLUMNS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _make_cmapss_id(subset: str, unit_id: int) -> str:
    return f"{CMAPSS_PREFIX}{subset}:{unit_id}"


def _parse_cmapss_id(source_id: str) -> tuple[str, int]:
    """Parse ``cmapss:FD001:1`` → (subset, unit_id)。"""
    body = source_id[len(CMAPSS_PREFIX):]
    parts = body.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid cmapss source ID: {source_id}（应为 cmapss:FD001:1）")
    subset = parts[0].upper()
    try:
        unit = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid cmapss unit id: {parts[1]!r}")
    return subset, unit


def _load_cmapss_engine(subset: str, unit_id: int) -> np.ndarray:
    """加载某台 engine 的退化传感器曲线（n_cycles, 14）。

    数据集路径默认 src/datasets/CMAPSSData/test_FD00X.txt。
    若文件不存在或 engine 不存在，抛 FileNotFoundError。
    """
    test_path = _CMAPSS_DATA_DIR / f"test_{subset}.txt"
    if not test_path.exists():
        raise FileNotFoundError(
            f"C-MAPSS test file not found: {test_path}\n"
            f"数据集需放在 datasets/CMAPSSData/ 下（手动备份）"
        )
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("CMapssSource 需要 pandas，请 pip install pandas") from e
    df = pd.read_csv(test_path, sep=r"\s+", header=None, names=_CMAPSS_COLUMNS)
    mask = df["unit"] == unit_id
    if not mask.any():
        raise ValueError(
            f"engine unit={unit_id} 不在 {subset}（可用 unit: "
            f"{sorted(df['unit'].unique())[:10]}...）"
        )
    sensor_cols = [f"sensor_{i}" for i in _CMAPSS_DEGRADATION_SENSORS]
    return df.loc[mask, sensor_cols].to_numpy(dtype=np.float32)


@register_source(CMAPSS_PREFIX)
class CMapssSource(SensorSource):
    """把 C-MAPSS 单台 engine 的退化曲线作为 1D 标量流送出。

    每次 ``read(n)`` 返回 n 个连续 cycle 的「退化指数」（14 维归一化均值），
    按 cycle 顺序推进。到末尾时按 ``loop`` 配置决定回卷或停止。

    与 FileSource 的区别：FileSource 重放真实 telemetry；CMapssSource 把研究
    用退化数据集投影成单维信号，目的是让特殊 RUL 传感器在天地链路上可见。
    """

    def __init__(
        self,
        subset: str = "FD001",
        unit_id: int = 1,
        sample_rate: float = 1.0,
        noise: SensorNoiseConfig | None = None,
        loop: bool = True,
    ):
        subset = subset.upper()
        if subset not in ("FD001", "FD002", "FD003", "FD004"):
            raise ValueError(f"Unsupported C-MAPSS subset: {subset}")
        self._subset = subset
        self._unit_id = int(unit_id)
        self._sample_rate = sample_rate
        self._noise = noise or SensorNoiseConfig()
        self._loop = loop
        self._source_id = _make_cmapss_id(subset, self._unit_id)
        self._channel = f"CMAPSS_{subset}_{self._unit_id}"

        # 加载并预计算 1D 退化信号
        raw_14d = _load_cmapss_engine(subset, self._unit_id)  # (T, 14)
        if raw_14d.shape[0] < 2:
            raise ValueError(f"engine {unit_id} 数据太少（{raw_14d.shape[0]} cycles）")
        # Z-score 归一化（用整条序列的均值/方差）
        mean = raw_14d.mean(axis=0, keepdims=True)
        std = raw_14d.std(axis=0, keepdims=True)
        std_safe = np.where(std > 1e-8, std, 1.0)
        z = (raw_14d - mean) / std_safe  # (T, 14)
        # 14 维均值 → 1D 退化曲线
        signal = z.mean(axis=1).astype(np.float32)
        # 进一步线性归一化到 [-1, 1]，便于与 C-1 / D-14 同图表显示
        s_max = float(np.abs(signal).max()) or 1.0
        self._signal = (signal / s_max).astype(np.float32)
        self._pos = 0
        self._exhausted = False

    def read(self, n: int) -> np.ndarray:
        if self._exhausted and not self._loop:
            return np.empty(0, dtype=np.float32)

        data_len = len(self._signal)
        if not self._loop:
            available = min(n, data_len - self._pos)
            if available == 0:
                self._exhausted = True
                return np.empty(0, dtype=np.float32)
            chunk = self._signal[self._pos:self._pos + available].copy()
            self._pos += available
            if self._pos >= data_len:
                self._exhausted = True
            if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
                chunk = _apply_noise(chunk, self._noise)
            return chunk

        # loop 模式
        result = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            chunk_size = min(n - filled, data_len - self._pos)
            result[filled:filled + chunk_size] = self._signal[self._pos:self._pos + chunk_size]
            filled += chunk_size
            self._pos += chunk_size
            if self._pos >= data_len:
                self._pos = 0
        if self._noise.missing_rate > 0 or self._noise.noise_std > 0:
            result = _apply_noise(result, self._noise)
        return result

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
    def source_id(self) -> str:
        return self._source_id

    @property
    def label(self) -> str:
        return f"🛰️ C-MAPSS · {self._subset} engine #{self._unit_id}"

    @property
    def is_virtual(self) -> bool:
        return False

    @classmethod
    def from_source_id(
        cls,
        source_id: str,
        *,
        sample_rate: float = 1.0,
        noise: SensorNoiseConfig | None = None,
        loop: bool = True,
        **_unused,
    ) -> "CMapssSource":
        """Parse ``cmapss:FD001:1`` and build a CMapssSource."""
        subset, unit_id = _parse_cmapss_id(source_id)
        return cls(
            subset=subset, unit_id=unit_id,
            sample_rate=sample_rate, noise=noise, loop=loop,
        )


# ---------------------------------------------------------------------------
# Global source registry
# ---------------------------------------------------------------------------

def list_all_sources(dataset_dir: str | None = None) -> list[dict]:
    """Return a unified list of ALL available data sources.

    Each entry::

        {
            "id": "file:NASA-MSL/C-1",
            "label": "📂 NASA-MSL · C-1",
            "is_virtual": False,
            "channel": "C-1",
            "dataset": "NASA-MSL",
        }

    Virtual sensors are always included; file sources are discovered from the
    datasets directory.
    """
    sources: list[dict] = []

    # ---- file sources ----
    for ds_name in ["NASA-MSL", "NASA-SMAP"]:
        try:
            channels = list_channels(ds_name)
        except Exception:
            continue
        for ch_name, _train_path, _test_path in channels:
            sources.append({
                "id": _make_file_id(ds_name, ch_name),
                "label": f"📂 {ds_name} · {ch_name}",
                "is_virtual": False,
                "dataset": ds_name,
                "channel": ch_name,
            })

    # ---- virtual sensors ----
    for sig_type, cfg in _VIRTUAL_PRESETS.items():
        sources.append({
            "id": _make_virtual_id(sig_type),
            "label": _VIRTUAL_LABELS.get(sig_type, f"🎛️ 虚拟传感器 · {sig_type}"),
            "is_virtual": True,
            "signal_type": sig_type,
        })

    # ---- C-MAPSS engine sources（特殊 RUL 演示传感器） ----
    # 列出每个 subset 的 unit=1 engine（演示用，不全列 100 台）。
    # 数据集不存在时静默跳过（datasets/CMAPSSData 是手动备份的大文件）。
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        test_path = _CMAPSS_DATA_DIR / f"test_{subset}.txt"
        if not test_path.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_csv(test_path, sep=r"\s+", header=None, names=_CMAPSS_COLUMNS)
            # 只列 unit=1（演示用）；如需扩展可改这里
            if (df["unit"] == 1).any():
                sources.append({
                    "id": _make_cmapss_id(subset, 1),
                    "label": f"🛰️ C-MAPSS · {subset} engine #1（特殊·退化演示）",
                    "is_virtual": False,
                    "dataset": f"C-MAPSS-{subset}",
                    "channel": f"CMAPSS_{subset}_1",
                    "is_special": True,
                })
        except Exception:
            # pandas 缺失或解析失败，跳过（不影响其他 source）
            continue

    return sources


# ---------------------------------------------------------------------------
# Factory (delegates to the registry populated above)
# ---------------------------------------------------------------------------

def create_source(
    source_id: str,
    sample_rate: float = 1.0,
    noise: SensorNoiseConfig | None = None,
    loop: bool = False,
    signal_freq_hz: float | None = None,
) -> SensorSource:
    """Factory: create a SensorSource from a source ID.

    Looks up the prefix in ``_SOURCE_REGISTRY`` and delegates to the
    matching class's ``from_source_id`` classmethod.  Unrecognised prefixes
    raise ``ValueError`` listing the registered prefixes.

    Args:
        source_id: e.g. ``"file:NASA-MSL/C-1"`` or ``"virtual:sine"``
        sample_rate: Hz for pacing; -1 for bulk mode (file sources only)
        noise: optional noise configuration
        loop: if True, FileSource rewinds on exhaustion (debugging)
        signal_freq_hz: for virtual sensors, signal frequency in Hz
                        (converted to cycles-per-sample internally)

    Returns:
        A ``SensorSource`` instance ready for ``read()``.
    """
    for prefix, cls in _SOURCE_REGISTRY.items():
        if source_id.startswith(prefix):
            return cls.from_source_id(
                source_id,
                sample_rate=sample_rate,
                noise=noise,
                loop=loop,
                signal_freq_hz=signal_freq_hz,
            )
    raise ValueError(
        f"Invalid source ID: {source_id!r}. "
        f"Registered prefixes: {registered_source_prefixes()}"
    )


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

DatasetSource = FileSource
SyntheticSource = VirtualSensorSource
