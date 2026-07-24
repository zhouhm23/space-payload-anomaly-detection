"""Unit tests for the modular cascade rule library.

Each test exercises a single rule module (one of the 9 under
``phm.algorithm.rules``) and asserts the rule's behaviour matches the
*original* monolithic ClassicFilter / PhysicalConstraint logic.  These
tests are the per-module regression guards for the Stage-1 structural
refactor — the cross-module aggregation equivalence is covered by
``test_rules_equivalence.py``.

The contract being verified:
  * Each rule is registered under its canonical name in FILTER_REGISTRY.
  * Each rule's decision / rule-name / per-sample-score contributions
    match what the original monolithic code produced for that rule.
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

from phm.algorithm.rules import (
    FILTER_REGISTRY,
    build_filter,
    DEFAULT_L1_MODULES,
    DEFAULT_L3_MODULES,
    L1ConstantRule,
    L1SigmaRule,
    L1IqrRule,
    L1RateRule,
    L3NanSanitiseRule,
    L3ConstantRule,
    L3RangeRule,
    L3RateRule,
    L3VarianceRule,
)
from phm.algorithm.cascade_types import (
    DECISION_PASS,
    DECISION_ALERT,
    DECISION_SKIP,
    DECISION_SUSPICIOUS,
)


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

class TestRegistry:
    """The registry must expose all rule modules by canonical name.

    Day26 registered the 9 L1/L3 rule modules; Day26-续 added the 3
    Layer 3.5 leak-free post-processing modules (knee / EMA / persistence),
    bringing the total to 12; Phase 1.5 added the opt-in ``l1_setpoint``
    expert module, bringing the total to 13.  This test guards all three
    additions.
    """

    def test_all_thirteen_modules_registered(self):
        # The 9 L1/L3 rules from Day26.
        l1_l3 = {
            "l1_constant", "l1_sigma", "l1_iqr", "l1_rate",
            "l3_nan_sanitise", "l3_constant", "l3_range",
            "l3_rate", "l3_variance",
        }
        # The 3 Layer 3.5 post-processing modules from Day26-续.
        l35 = {"l3_knee_threshold", "l3_ema_smoothing", "l3_persistence"}
        # The opt-in Phase 1.5 expert module (NOT in DEFAULT_L1_MODULES).
        l1_expert = {"l1_setpoint"}
        assert set(FILTER_REGISTRY.keys()) == (l1_l3 | l35 | l1_expert)

    def test_l1_l3_subset_still_registered(self):
        """The original 9 L1/L3 modules must remain registered unchanged
        even as new Layer 3.5 modules are added (forward-compat guard)."""
        l1_l3 = {
            "l1_constant", "l1_sigma", "l1_iqr", "l1_rate",
            "l3_nan_sanitise", "l3_constant", "l3_range",
            "l3_rate", "l3_variance",
        }
        assert l1_l3.issubset(FILTER_REGISTRY.keys())

    def test_default_chains_match_canonical_order(self):
        # DEFAULT_L1_MODULES order is load-bearing for severity aggregation
        # (σ/IQR → alert has priority over rate → suspicious because they
        # appear earlier in the chain).
        assert DEFAULT_L1_MODULES == ["l1_constant", "l1_sigma", "l1_iqr", "l1_rate"]
        assert DEFAULT_L3_MODULES == [
            "l3_nan_sanitise", "l3_constant", "l3_range",
            "l3_rate", "l3_variance",
        ]

    @pytest.mark.parametrize("name", [
        "l1_constant", "l1_sigma", "l1_iqr", "l1_rate",
        "l3_nan_sanitise", "l3_constant", "l3_range",
        "l3_rate", "l3_variance",
    ])
    def test_build_filter_round_trip(self, name):
        """build_filter must return a BaseFilter whose name attr matches."""
        f = build_filter(name)
        assert f.name == name

    def test_build_filter_unknown_name_raises(self):
        with pytest.raises(KeyError):
            build_filter("does_not_exist")


# ---------------------------------------------------------------------------
# L1 rules
# ---------------------------------------------------------------------------

class TestL1ConstantRule:
    """L1ConstantRule owns empty/constant/insufficient-finite short-circuits."""

    def test_empty_input_skip(self):
        r = L1ConstantRule().filter(np.array([]))
        assert r.decision == DECISION_SKIP
        assert r.detail["reason"] == "empty_input"

    def test_constant_channel_skip(self):
        r = L1ConstantRule().filter(np.full(100, -1.0, dtype=np.float32))
        assert r.decision == DECISION_SKIP
        assert r.detail["rules"] == ["constant_channel"]
        assert r.detail["std"] < r.detail["threshold"]

    def test_near_constant_skip_with_custom_threshold(self):
        # Original ClassicFilter(constant_std=0.01) triggered on this data.
        values = np.full(100, 5.0, dtype=np.float32)
        values[0] += 0.001
        r = L1ConstantRule(constant_std=0.01).filter(values)
        assert r.decision == DECISION_SKIP

    def test_all_nan_insufficient_finite_skip(self):
        r = L1ConstantRule().filter(np.full(50, np.nan))
        assert r.decision == DECISION_SKIP
        assert r.detail["rules"] == ["insufficient_finite"]

    def test_normal_data_passes(self):
        # Normal Gaussian data must not short-circuit — the rule returns
        # pass with an all-zero per_sample_score.
        r = L1ConstantRule().filter(np.random.RandomState(42).randn(200).astype(np.float32))
        assert r.decision == DECISION_PASS
        assert r.detail["rules"] == []
        assert np.all(r.detail["per_sample_score"] == 0.0)

    def test_enable_constant_false_skips_std_check_only(self):
        # When enable_constant=False the std check is bypassed but the
        # insufficient-finite guard still runs.  A constant channel with
        # enough finite samples must therefore pass (matching the original
        # ClassicFilter(enable_constant=False) behaviour).
        r = L1ConstantRule(enable_constant=False).filter(np.full(100, 5.0, dtype=np.float32))
        assert r.decision == DECISION_PASS


class TestL1SigmaRule:
    """L1SigmaRule flags 3σ outliers with per-sample score 0.8."""

    def test_sigma_outlier_alert(self):
        values = np.random.RandomState(42).randn(1000).astype(np.float32)
        values[500] = 100.0
        r = L1SigmaRule(sigma_k=3.0).filter(values)
        assert r.decision == DECISION_ALERT
        assert r.detail["rules"] == ["sigma_3"]
        # Outlier sample must have been flagged.
        assert r.detail["per_sample_score"][500] == pytest.approx(0.8)

    def test_normal_data_no_trigger(self):
        # With sigma_k=3 on 1000 Gaussian samples, ~3 samples are expected
        # beyond 3σ — so the rule may or may not trigger.  Use a high k
        # to guarantee no trigger and verify pass + empty rules.
        values = np.random.RandomState(42).randn(200).astype(np.float32)
        r = L1SigmaRule(sigma_k=10.0).filter(values)
        assert r.decision == DECISION_PASS
        assert r.detail["rules"] == []

    def test_inf_not_flagged_op_precedence(self):
        """Regression: +Inf must NOT be flagged by the σ rule.

        ``finite_mask & (v < lo) | (v > hi)`` would parse as
        ``(finite_mask & (v < lo)) | (v > hi)`` and incorrectly flag +Inf.
        Parentheses are required around the OR.
        """
        values = np.random.RandomState(42).randn(200).astype(np.float32)
        values[10] = np.inf
        values[20] = -np.inf
        values[30] = np.nan
        values[50] = 50.0
        values[60] = -50.0
        r = L1SigmaRule(sigma_k=3.0).filter(values)
        ps = r.detail["per_sample_score"]
        assert ps[10] == 0.0, "+Inf should not be flagged"
        assert ps[20] == 0.0, "-Inf should not be flagged"
        assert ps[30] == 0.0, "NaN should not be flagged"
        assert ps[50] > 0.0, "real positive outlier must be flagged"
        assert ps[60] > 0.0, "real negative outlier must be flagged"

    def test_empty_input_passes(self):
        # The σ rule alone doesn't own the empty-input short-circuit —
        # that's L1ConstantRule's job.  σ must pass gracefully.
        r = L1SigmaRule().filter(np.array([]))
        assert r.decision == DECISION_PASS
        assert r.detail["rules"] == []


class TestL1IqrRule:
    """L1IqrRule flags Tukey-fence outliers with per-sample score 0.7."""

    def test_iqr_outlier_alert(self):
        values = np.random.RandomState(42).randn(100).astype(np.float32)
        values[50] = 50.0
        r = L1IqrRule(iqr_factor=1.5).filter(values)
        assert r.decision == DECISION_ALERT
        assert r.detail["rules"] == ["iqr"]
        assert r.detail["per_sample_score"][50] == pytest.approx(0.7)

    def test_insufficient_finite_passes(self):
        # IQR needs ≥4 finite samples.  Fewer → no trigger.
        r = L1IqrRule().filter(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert r.decision == DECISION_PASS
        assert r.detail["rules"] == []

    def test_inf_not_flagged_op_precedence(self):
        """Same operator-precedence regression guard as the σ rule."""
        values = np.random.RandomState(7).randn(200).astype(np.float32)
        values[10] = np.inf
        values[20] = -np.inf
        values[50] = 50.0
        r = L1IqrRule().filter(values)
        ps = r.detail["per_sample_score"]
        assert ps[10] == 0.0, "+Inf should not be flagged by IQR"
        assert ps[20] == 0.0, "-Inf should not be flagged by IQR"
        assert ps[50] > 0.0, "real outlier must be flagged by IQR"


class TestL1RateRule:
    """L1RateRule flags consecutive-sample jumps with per-sample score 0.6."""

    def test_rate_jump_suspicious_with_explicit_max_rate(self):
        values = np.array([0.0, 0.1, 0.0, 10.0, 0.0], dtype=np.float32)
        r = L1RateRule(max_rate=1.0).filter(values)
        assert r.decision == DECISION_SUSPICIOUS
        assert r.detail["rules"] == ["rate_of_change"]
        # Jump is detected on the *destination* sample (index 3).
        assert r.detail["per_sample_score"][3] == pytest.approx(0.6)

    def test_rate_no_trigger_on_smooth_data(self):
        values = np.linspace(0, 1, 100, dtype=np.float32)
        r = L1RateRule(max_rate=1.0).filter(values)
        assert r.decision == DECISION_PASS
        assert r.detail["rules"] == []

    def test_rate_derived_threshold_from_quantile(self):
        # When max_rate=None, threshold = p99(diff) * multiplier.
        # Construct data with a baseline of small normal variation plus one
        # dominant jump so p99(diff) > 0 and the spike exceeds p99*5.
        rng = np.random.RandomState(0)
        values = rng.randn(200).astype(np.float32) * 0.1  # small variation
        values[100] = 50.0  # dominant single jump → diff[99] is large
        r = L1RateRule(rate_quantile=99.0, rate_multiplier=5.0).filter(values)
        assert r.decision == DECISION_SUSPICIOUS
        assert "rate_of_change" in r.detail["rules"]


# ---------------------------------------------------------------------------
# L3 rules
# ---------------------------------------------------------------------------

class TestL3NanSanitiseRule:
    """L3NanSanitiseRule zeros scores at non-finite input positions."""

    def test_nan_zeroed(self):
        values = np.array([1.0, np.nan, 2.0, 3.0], dtype=np.float32)
        scores = np.array([0.5, 0.9, 0.3, 0.2], dtype=np.float32)
        r = L3NanSanitiseRule().filter(values, scores)
        adj = r.detail["adjusted_scores"]
        assert adj[1] == 0.0
        assert "nan_inf_sanitise" in r.detail["rules"]
        # Other positions preserved.
        assert adj[0] == pytest.approx(0.5)

    def test_no_trigger_on_clean_data(self):
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        r = L3NanSanitiseRule().filter(values, scores)
        assert r.detail["rules"] == []

    def test_scores_none_zeros(self):
        r = L3NanSanitiseRule().filter(np.array([1.0, np.nan], dtype=np.float32), None)
        assert np.all(r.detail["adjusted_scores"] == 0.0)


class TestL3ConstantRule:
    """L3ConstantRule suppresses scores on near-constant windows."""

    def test_constant_channel_all_zeroed(self):
        values = np.full(100, -1.0, dtype=np.float32)
        scores = np.full(100, 0.8, dtype=np.float32)
        r = L3ConstantRule().filter(values, scores)
        adj = r.detail["adjusted_scores"]
        assert np.all(adj == 0.0)
        assert "constant_suppression" in r.detail["rules"]
        # Stop-chain signal must be present so PhysicalConstraint skips
        # rules 3/4/5 (reproducing the original early-return).
        assert r.detail.get("_stop_chain") is True

    def test_normal_data_no_trigger(self):
        values = np.random.RandomState(42).randn(200).astype(np.float32)
        scores = np.full(200, 0.5, dtype=np.float32)
        r = L3ConstantRule().filter(values, scores)
        assert "constant_suppression" not in r.detail["rules"]
        assert r.detail.get("_stop_chain") is None


class TestL3RangeRule:
    """L3RangeRule boosts scores for out-of-range values."""

    def test_out_of_range_boosted(self):
        values = np.array([0.5, -0.3, 5.0, 0.1], dtype=np.float32)
        scores = np.array([0.1, 0.2, 0.05, 0.3], dtype=np.float32)
        r = L3RangeRule(valid_min=-1.0, valid_max=1.0, range_boost=0.95).filter(values, scores)
        adj = r.detail["adjusted_scores"]
        assert adj[2] >= 0.95
        assert "range_boundary" in r.detail["rules"]

    def test_no_bounds_is_noop(self):
        values = np.array([0.5, 5.0], dtype=np.float32)
        scores = np.array([0.1, 0.2], dtype=np.float32)
        r = L3RangeRule().filter(values, scores)  # valid_min/max both None
        assert r.detail["rules"] == []
        # Scores passed through unchanged.
        assert np.allclose(r.detail["adjusted_scores"], scores)


class TestL3RateRule:
    """L3RateRule boosts scores where consecutive jumps exceed max_rate."""

    def test_large_jump_boosted(self):
        values = np.array([0.0, 0.1, 0.0, 10.0, 0.0], dtype=np.float32)
        scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        r = L3RateRule(max_rate=1.0, rate_boost=0.85).filter(values, scores)
        adj = r.detail["adjusted_scores"]
        assert adj[3] >= 0.85
        assert "rate_ceiling" in r.detail["rules"]

    def test_no_max_rate_is_noop(self):
        values = np.array([0.0, 100.0], dtype=np.float32)
        scores = np.array([0.1, 0.1], dtype=np.float32)
        r = L3RateRule().filter(values, scores)  # max_rate=None
        assert r.detail["rules"] == []


class TestL3VarianceRule:
    """L3VarianceRule dampens all scores when window variance drifts."""

    def test_drift_dampen_triggered(self):
        # baseline_var=0.01, ratio=10 → window_var > 0.1 triggers.
        values = np.random.RandomState(0).randn(200).astype(np.float32) * 5
        scores = np.full(200, 0.8, dtype=np.float32)
        r = L3VarianceRule(
            baseline_var=0.01, var_dampen_ratio=10.0, var_dampen_factor=0.3,
        ).filter(values, scores)
        assert "variance_drift_dampen" in r.detail["rules"]
        adj = r.detail["adjusted_scores"]
        assert np.all(adj <= 0.8 * 0.3 + 1e-6)

    def test_no_baseline_is_noop(self):
        values = np.random.RandomState(0).randn(100).astype(np.float32) * 5
        scores = np.full(100, 0.8, dtype=np.float32)
        r = L3VarianceRule().filter(values, scores)  # baseline_var=None
        assert r.detail["rules"] == []

    def test_low_variance_no_trigger(self):
        # Window variance below the dampen ratio must not trigger.
        values = np.random.RandomState(0).randn(200).astype(np.float32) * 0.01
        scores = np.full(200, 0.8, dtype=np.float32)
        r = L3VarianceRule(
            baseline_var=1.0, var_dampen_ratio=10.0, var_dampen_factor=0.3,
        ).filter(values, scores)
        assert "variance_drift_dampen" not in r.detail["rules"]
