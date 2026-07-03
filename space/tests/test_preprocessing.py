"""Unit tests for space.preprocessing — on-orbit preprocessing pipeline."""

import numpy as np
import pytest

from preprocessing import SpacePreprocessor


class TestSpacePreprocessor:
    """Tests for the space-segment preprocessing pipeline."""

    def test_fit_transform_shapes(self):
        ts = np.random.randn(500).astype(np.float32)
        proc = SpacePreprocessor()
        out = proc.fit_transform(ts)
        assert out.shape == ts.shape
        assert out.dtype == np.float32

    def test_imputes_nan(self):
        """NaN values should be filled after transform."""
        ts = np.random.randn(500).astype(np.float32)
        raw = ts.copy()
        raw[50:60] = np.nan
        raw[200] = np.nan
        assert np.isnan(raw).any()
        proc = SpacePreprocessor()
        out = proc.fit_transform(raw, train_values=ts)
        assert not np.isnan(out).any(), "Output should have no NaN"

    def test_normalization_zero_mean(self):
        """When fit on the same data, output mean ≈ 0."""
        ts = np.random.randn(2000).astype(np.float32) * 5 + 3
        proc = SpacePreprocessor()
        out = proc.fit_transform(ts)
        assert abs(out.mean()) < 0.1, f"Mean should be ~0, got {out.mean()}"

    def test_transform_without_fit_raises(self):
        proc = SpacePreprocessor()
        with pytest.raises(RuntimeError):
            proc.transform(np.random.randn(100))

    def test_end_to_end_with_source(self):
        """Integration: source → preprocess → no NaN."""
        from sensor_source import (
            SyntheticSource, SyntheticConfig, SensorNoiseConfig,
        )
        noise = SensorNoiseConfig(missing_rate=0.1, noise_std=0.1, random_seed=42)
        src = SyntheticSource(
            config=SyntheticConfig(signal_type="sine", frequency=0.02),
            noise=noise,
        )
        raw = src.read(512)
        assert np.isnan(raw).any(), "Source should have NaN from noise"
        proc = SpacePreprocessor()
        cleaned = proc.fit_transform(raw)
        assert not np.isnan(cleaned).any()
        assert len(cleaned) == 512
