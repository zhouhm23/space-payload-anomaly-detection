"""Unit tests for interface.sensor_source — simulated DAQ card.

These tests verify both DatasetSource and SyntheticSource without requiring
any model loading, so they run fast.
"""

import numpy as np
import pytest

import os, sys
_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sensor_source import (
    DatasetSource, SyntheticSource, SyntheticConfig, SensorNoiseConfig,
)


class TestDatasetSource:
    """Tests for DatasetSource (NASA data replay)."""

    def test_reads_data(self):
        src = DatasetSource(dataset="NASA-MSL", channel="C-1")
        chunk = src.read(512)
        assert len(chunk) == 512
        assert chunk.dtype == np.float32
        assert not src.exhausted

    def test_exhausts_and_returns_empty(self):
        """After reading past the end, should return empty array, not zeros."""
        src = DatasetSource(dataset="NASA-MSL", channel="C-1")
        big = src.read(100000)
        assert src.exhausted
        extra = src.read(100)
        assert len(extra) == 0

    def test_reset(self):
        """Reset should rewind the source."""
        src = DatasetSource(dataset="NASA-MSL", channel="C-1")
        a = src.read(100)
        src.reset()
        b = src.read(100)
        np.testing.assert_array_equal(a, b)

    def test_channel_name(self):
        src = DatasetSource(dataset="NASA-MSL", channel="C-1")
        assert src.channel_name == "C-1"

    def test_default_channel(self):
        """If no channel specified, should pick the first one."""
        src = DatasetSource(dataset="NASA-MSL")
        assert src.channel_name is not None

    def test_invalid_channel_raises(self):
        with pytest.raises(ValueError):
            DatasetSource(dataset="NASA-MSL", channel="NONEXISTENT")

    def test_noise_injection(self):
        """Noise config should produce NaN for missing values."""
        noise = SensorNoiseConfig(missing_rate=0.1, noise_std=0.05, random_seed=42)
        src = DatasetSource(dataset="NASA-MSL", channel="C-1", noise=noise)
        chunk = src.read(512)
        assert np.isnan(chunk).any(), "Should have NaN from missing_rate"

    def test_clean_source_no_nan(self):
        """Without noise, no NaN should appear."""
        src = DatasetSource(dataset="NASA-MSL", channel="C-1")
        chunk = src.read(512)
        assert not np.isnan(chunk).any()


class TestSyntheticSource:
    """Tests for SyntheticSource (continuous signal generator)."""

    def test_sine_signal(self):
        cfg = SyntheticConfig(signal_type="sine", frequency=0.1, amplitude=1.0)
        src = SyntheticSource(config=cfg)
        chunk = src.read(256)
        assert len(chunk) == 256
        assert chunk.dtype == np.float32
        # Sine wave should have values near ±amplitude
        assert np.abs(chunk).max() > 0.5

    def test_never_exhausts(self):
        src = SyntheticSource()
        for _ in range(100):
            src.read(512)
        assert not src.exhausted

    def test_multi_sine(self):
        cfg = SyntheticConfig(signal_type="multi_sine", frequency=0.02)
        src = SyntheticSource(config=cfg)
        chunk = src.read(512)
        # Multi-sine should have non-trivial variance
        assert np.std(chunk) > 0.01

    def test_continuity_across_reads(self):
        """Signal should be continuous across read() calls (no reset)."""
        cfg = SyntheticConfig(signal_type="sine", frequency=0.05)
        src = SyntheticSource(config=cfg)
        a = src.read(100)
        b = src.read(100)
        combined = np.concatenate([a, b])
        # Generate the same signal in one call
        src2 = SyntheticSource(config=cfg)
        full = src2.read(200)
        np.testing.assert_allclose(combined, full, atol=1e-5)

    def test_channel_name(self):
        src = SyntheticSource(config=SyntheticConfig(signal_type="sine"))
        assert "sine" in src.channel_name

    def test_anomaly_injection(self):
        """Anomaly spikes should increase the signal range."""
        cfg_clean = SyntheticConfig(signal_type="sine", frequency=0.02, amplitude=1.0)
        cfg_spiky = SyntheticConfig(
            signal_type="sine", frequency=0.02, amplitude=1.0,
            anomaly_every=100, anomaly_magnitude=5.0,
        )
        src_clean = SyntheticSource(config=cfg_clean)
        src_spiky = SyntheticSource(config=cfg_spiky)
        clean = src_clean.read(500)
        spiky = src_spiky.read(500)
        assert spiky.max() > clean.max() + 2.0

    def test_noise_on_synthetic(self):
        """Noise config should work on synthetic source too."""
        noise = SensorNoiseConfig(noise_std=0.1, random_seed=42)
        cfg = SyntheticConfig(signal_type="sine", frequency=0.02)
        src = SyntheticSource(config=cfg, noise=noise)
        chunk = src.read(256)
        assert len(chunk) == 256


class TestCMapssSource:
    """Tests for CMapssSource (C-MAPSS engine degradation replay).

    需要本地 datasets/CMAPSSData/test_FD001.txt（手动备份）。若不存在则
    skip——避免 CI 没数据时挂掉。
    """

    def _check_dataset(self):
        from pathlib import Path
        # __file__ = 生产实习/src/space/tests/test_sensor_source.py
        # 上 4 次 parent → 生产实习/，然后 datasets/CMAPSSData
        data_dir = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "CMAPSSData"
        if not (data_dir / "test_FD001.txt").exists():
            pytest.skip("C-MAPSS dataset not available (manual backup required)")

    def test_create_and_read(self):
        self._check_dataset()
        from sensor_source import create_source
        src = create_source("cmapss:FD001:1", sample_rate=1.0, loop=True)
        assert src.channel_name == "CMAPSS_FD001_1"
        assert src.source_id == "cmapss:FD001:1"
        chunk = src.read(10)
        assert chunk.dtype == np.float32
        assert chunk.shape == (10,)
        # 归一化到 [-1, 1]
        assert chunk.min() >= -1.01 and chunk.max() <= 1.01

    def test_loop_rewinds(self):
        """loop=True 时读超过数据长度应该回卷（不返回空）。"""
        self._check_dataset()
        from sensor_source import create_source
        src = create_source("cmapss:FD001:1", loop=True)
        # engine 1 在 test_FD001 有 31 cycles，读 100 个点应自动回卷
        chunk = src.read(100)
        assert chunk.shape == (100,)
        assert src.exhausted is False

    def test_non_loop_exhausts(self):
        """loop=False 时读超过数据长度后 exhausted=True。"""
        self._check_dataset()
        from sensor_source import create_source
        src = create_source("cmapss:FD001:1", loop=False)
        # 先读 1000 个（远超 31 cycles）
        chunk = src.read(1000)
        # 实际只返回 31 个，然后 exhausted
        assert chunk.shape[0] <= 31
        assert src.exhausted is True
        # 再读返回空
        chunk2 = src.read(10)
        assert chunk2.shape == (0,)

    def test_invalid_subset_raises(self):
        from sensor_source import create_source
        with pytest.raises(ValueError, match="Unsupported C-MAPSS subset|Invalid"):
            create_source("cmapss:FD999:1", loop=True)

    def test_invalid_unit_raises(self):
        self._check_dataset()
        from sensor_source import create_source
        with pytest.raises(ValueError):
            create_source("cmapss:FD001:99999", loop=True)

    def test_invalid_id_format_raises(self):
        from sensor_source import create_source
        with pytest.raises(ValueError):
            create_source("cmapss:FD001", loop=True)  # 缺 unit id

    def test_list_all_sources_includes_cmapss(self):
        """list_all_sources 应包含 cmapss sources。"""
        self._check_dataset()
        from sensor_source import list_all_sources
        all_sources = list_all_sources()
        cmapss_ids = [s['id'] for s in all_sources if s['id'].startswith('cmapss:')]
        assert "cmapss:FD001:1" in cmapss_ids
