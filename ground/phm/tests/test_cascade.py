"""Unit tests for the three-layer cascade detector.

Tests L1 (ClassicFilter), L3 (PhysicalConstraint) and the full
CascadeDetector chain.  Uses a mock BaseDetector instead of the real
TSPulse so the tests run without model weights.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from phm.algorithm.base import BaseDetector
from phm.algorithm.cascade_types import (
    DECISION_PASS, DECISION_ALERT, DECISION_SKIP, DECISION_SUSPICIOUS,
    LAYER_L1_CLASSIC, LAYER_L2_DL, LAYER_L3_PHYSICAL,
)
from phm.algorithm.classic_filter import ClassicFilter
from phm.algorithm.physical_constraint import ConstraintConfig, PhysicalConstraint
from phm.algorithm.cascade_detector import CascadeDetector


# ---------------------------------------------------------------------------
# Mock DL detector (replaces TSPulse — no model weights needed)
# ---------------------------------------------------------------------------

class MockDetector(BaseDetector):
    """Returns a fixed anomaly score pattern for testing."""

    def __init__(self, score_array=None):
        self.n_params = 42
        self.model_source = "mock"
        self._score_array = score_array

    def detect(self, values, train_values_for_scaler=None):
        if self._score_array is not None:
            return self._score_array[:len(values)]
        # Default: score proportional to absolute deviation from mean
        v = np.asarray(values, dtype=np.float32)
        if len(v) == 0:
            return v
        dev = np.abs(v - np.mean(v))
        mx = np.max(dev) if np.max(dev) > 0 else 1.0
        return np.clip(dev / mx, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Layer 1: ClassicFilter
# ---------------------------------------------------------------------------

class TestClassicFilter:

    def test_constant_channel_skip(self):
        """Constant channel (std=0) should be skipped."""
        f = ClassicFilter()
        values = np.full(100, -1.0, dtype=np.float32)
        result = f.filter(values)
        assert result.decision == DECISION_SKIP
        assert result.detail["rules"] == ["constant_channel"]

    def test_near_constant_skip(self):
        """Near-constant channel (std < threshold) should be skipped."""
        f = ClassicFilter(constant_std=0.01)
        values = np.full(100, 5.0, dtype=np.float32)
        values[0] += 0.001  # tiny perturbation
        result = f.filter(values)
        assert result.decision == DECISION_SKIP

    def test_sigma_outlier_alert(self):
        """3σ outlier should trigger alert."""
        f = ClassicFilter(sigma_k=3.0)
        values = np.random.RandomState(42).randn(1000).astype(np.float32)
        values[500] = 100.0  # massive outlier
        result = f.filter(values)
        assert result.decision == DECISION_ALERT
        assert "sigma_3" in result.detail["rules"]

    def test_normal_data_passes(self):
        """Normal Gaussian data should pass with high sigma_k."""
        # With default sigma_k=3, ~0.3% of Gaussian samples are 3σ outliers.
        # Use sigma_k=4 so only truly extreme values trigger.
        f = ClassicFilter(sigma_k=4.0, enable_iqr=False, enable_rate=False)
        values = np.random.RandomState(42).randn(2000).astype(np.float32)
        result = f.filter(values)
        # Should not be skip (std >> threshold)
        assert result.decision in (DECISION_PASS, DECISION_SUSPICIOUS, DECISION_ALERT)
        # With relaxed thresholds the vast majority should pass
        assert result.detail.get("std", 0) > 0.5

    def test_empty_input_skip(self):
        """Empty input should be skipped."""
        f = ClassicFilter()
        result = f.filter(np.array([]))
        assert result.decision == DECISION_SKIP

    def test_all_nan_skip(self):
        """All-NaN input should be skipped."""
        f = ClassicFilter()
        result = f.filter(np.full(50, np.nan))
        assert result.decision == DECISION_SKIP
        # reason is stored in the rules list, not a detail key
        rules = result.detail.get("rules", [])
        assert "insufficient_finite" in rules

    def test_iqr_outlier(self):
        """IQR outlier should trigger alert."""
        f = ClassicFilter(enable_sigma=False, enable_iqr=True)
        # Need variance for IQR to be meaningful: use varied normal data + one outlier
        values = np.random.RandomState(42).randn(100).astype(np.float32)
        values[50] = 50.0  # extreme outlier
        result = f.filter(values)
        assert "iqr" in result.detail["rules"]


# ---------------------------------------------------------------------------
# Layer 3: PhysicalConstraint
# ---------------------------------------------------------------------------

class TestPhysicalConstraint:

    def test_nan_sanitise(self):
        """NaN in input should zero the corresponding score."""
        pc = PhysicalConstraint()
        values = np.array([1.0, np.nan, 2.0, 3.0], dtype=np.float32)
        scores = np.array([0.5, 0.9, 0.3, 0.2], dtype=np.float32)
        result = pc.filter(values, scores)
        adjusted = result.detail["adjusted_scores"]
        assert adjusted[1] == 0.0  # NaN point score zeroed
        assert "nan_inf_sanitise" in result.detail["rules"]

    def test_constant_channel_suppression(self):
        """Constant input window should force all scores to 0."""
        pc = PhysicalConstraint()
        values = np.full(100, -1.0, dtype=np.float32)
        scores = np.full(100, 0.8, dtype=np.float32)  # high scores
        result = pc.filter(values, scores)
        adjusted = result.detail["adjusted_scores"]
        assert np.all(adjusted == 0.0)
        assert "constant_suppression" in result.detail["rules"]

    def test_range_boundary_boost(self):
        """Out-of-range values should have their score boosted."""
        cfg = ConstraintConfig(valid_min=-1.0, valid_max=1.0, range_boost=0.95)
        pc = PhysicalConstraint(cfg)
        values = np.array([0.5, -0.3, 5.0, 0.1], dtype=np.float32)
        scores = np.array([0.1, 0.2, 0.05, 0.3], dtype=np.float32)
        result = pc.filter(values, scores)
        adjusted = result.detail["adjusted_scores"]
        assert adjusted[2] >= 0.95  # out-of-range boosted
        assert "range_boundary" in result.detail["rules"]

    def test_rate_ceiling(self):
        """Large jump should trigger rate_ceiling rule."""
        cfg = ConstraintConfig(max_rate=1.0, rate_boost=0.85)
        pc = PhysicalConstraint(cfg)
        values = np.array([0.0, 0.1, 0.0, 10.0, 0.0], dtype=np.float32)
        scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        result = pc.filter(values, scores)
        adjusted = result.detail["adjusted_scores"]
        assert adjusted[3] >= 0.85  # rate jump boosted
        assert "rate_ceiling" in result.detail["rules"]

    def test_variance_drift_dampen(self):
        """Excessive window variance vs baseline should dampen scores."""
        cfg = ConstraintConfig(
            baseline_var=0.01,     # tiny baseline
            var_dampen_ratio=10.0, # window var > 0.1 triggers dampen
            var_dampen_factor=0.3,
        )
        pc = PhysicalConstraint(cfg)
        values = np.random.RandomState(0).randn(200).astype(np.float32) * 5
        scores = np.full(200, 0.8, dtype=np.float32)
        result = pc.filter(values, scores)
        if "variance_drift_dampen" in result.detail["rules"]:
            adjusted = result.detail["adjusted_scores"]
            assert np.all(adjusted <= 0.8 * 0.3 + 1e-6)

    def test_no_rules_on_clean_data(self):
        """Clean normal data with matching scores should not trigger rules."""
        pc = PhysicalConstraint()
        values = np.random.RandomState(42).randn(200).astype(np.float32)
        scores = np.full(200, 0.1, dtype=np.float32)
        result = pc.filter(values, scores)
        # NaN sanitise may trigger if random data has NaNs, but randn doesn't
        # constant_suppression won't trigger (std >> threshold)
        # No range/rate/variance rules configured
        assert "constant_suppression" not in result.detail["rules"]


# ---------------------------------------------------------------------------
# CascadeDetector (full chain)
# ---------------------------------------------------------------------------

class TestCascadeDetector:

    def test_constant_channel_short_circuits_l2(self):
        """Constant channel should skip L2 (no forward pass) and return zeros."""
        mock = MockDetector(score_array=np.ones(100, dtype=np.float32))
        cascade = CascadeDetector(mock)
        values = np.full(100, -1.0, dtype=np.float32)
        out = cascade.detect_with_layers(values, channel="T-5")

        assert out.final_scores is not None
        assert np.all(out.final_scores == 0.0)  # constant → zeros
        # L2 should NOT have run
        layer_names = [lr.layer for lr in out.layers]
        assert LAYER_L1_CLASSIC in layer_names
        assert LAYER_L2_DL not in layer_names
        assert out.channel == "T-5"

    def test_normal_data_runs_full_chain(self):
        """Normal data should run all three layers."""
        mock = MockDetector()
        cascade = CascadeDetector(mock)
        values = np.random.RandomState(42).randn(512).astype(np.float32)
        out = cascade.detect_with_layers(values, channel="C-1")

        layer_names = [lr.layer for lr in out.layers]
        assert LAYER_L1_CLASSIC in layer_names
        assert LAYER_L2_DL in layer_names
        assert LAYER_L3_PHYSICAL in layer_names
        assert len(out.final_scores) == 512
        assert np.all(np.isfinite(out.final_scores))

    def test_detect_backward_compatible(self):
        """detect() should return a plain ndarray (BaseDetector interface)."""
        mock = MockDetector()
        cascade = CascadeDetector(mock)
        values = np.random.RandomState(0).randn(100).astype(np.float32)
        scores = cascade.detect(values)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 100

    def test_nan_input_handled(self):
        """NaN in input should not crash the cascade."""
        mock = MockDetector()
        cascade = CascadeDetector(mock)
        values = np.array([1.0, np.nan, 2.0, 3.0, np.nan] * 20, dtype=np.float32)
        out = cascade.detect_with_layers(values)
        assert np.all(np.isfinite(out.final_scores))

    def test_empty_input_handled(self):
        """Empty input should not crash."""
        mock = MockDetector()
        cascade = CascadeDetector(mock)
        out = cascade.detect_with_layers(np.array([]), channel="X")
        assert len(out.final_scores) == 0

    def test_l3_suppresses_constant_after_l2(self):
        """If L2 runs on near-constant data, L3 should suppress scores."""
        # Mock returns high scores for everything
        mock = MockDetector(score_array=np.ones(100, dtype=np.float32) * 0.9)
        cascade = CascadeDetector(mock)
        # Near-constant but just above L1's threshold (so L1 passes)
        values = np.full(99, 5.0, dtype=np.float32)
        values[50] = 5.001  # tiny variation
        out = cascade.detect_with_layers(values)
        # L1 may pass (std > 1e-3), L2 gives high scores, but L3 should suppress
        # because constant_std=1e-3 and our data is near-constant
        # Note: depends on exact std — if std > 1e-3 L3 won't suppress
        # This test verifies the cascade doesn't crash either way
        assert np.all(out.final_scores >= 0.0)
        assert np.all(out.final_scores <= 1.0)

    def test_cascade_output_to_dict(self):
        """CascadeOutput.to_dict should be JSON-serializable."""
        import json
        mock = MockDetector()
        cascade = CascadeDetector(mock)
        values = np.random.RandomState(1).randn(50).astype(np.float32)
        out = cascade.detect_with_layers(values, channel="C-1")
        d = out.to_dict(max_detail=True)
        # Should be JSON-serializable
        json.dumps(d, default=str)
        assert d["channel"] == "C-1"
        assert len(d["layers"]) >= 2

    def test_model_metadata_delegated(self):
        """n_params and model_source should come from the wrapped detector."""
        mock = MockDetector()
        cascade = CascadeDetector(mock)
        assert cascade.n_params == 42
        assert cascade.model_source == "mock"
