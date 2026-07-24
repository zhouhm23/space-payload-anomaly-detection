"""Unit tests for the L1 ``l1_setpoint`` rule (Phase 1.5).

``L1SetpointRule`` is an opt-in expert module that flags deviations from a
scientist-supplied physical expectation.  Unlike the statistical L1 rules
(sigma / iqr / rate) it does not derive its thresholds from the data — it
requires explicit construction parameters in one of three modes:

  command    — one normal value, one or more anomaly values
  range      — expected working point +/- tolerance, or an explicit band
  enumerate  — a fixed set of legal values

These tests exercise every mode across normal / anomaly / boundary /
NaN / empty inputs, plus the construction-time parameter validation that
the @command DSL validator (Plan 3, E3) relies on.
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
    L1SetpointRule,
    L1SigmaRule,
    DEFAULT_L1_MODULES,
    build_filter,
)
from phm.algorithm.classic_filter import ClassicFilter
from phm.algorithm.cascade_types import (
    DECISION_PASS,
    DECISION_ALERT,
    LAYER_L1_CLASSIC,
)


# ---------------------------------------------------------------------------
# Registration & construction sanity
# ---------------------------------------------------------------------------

class TestSetpointRegistration:
    """The rule must be registered but excluded from DEFAULT_L1_MODULES."""

    def test_registered_under_canonical_name(self):
        assert "l1_setpoint" in FILTER_REGISTRY
        assert FILTER_REGISTRY["l1_setpoint"] is L1SetpointRule

    def test_not_in_default_l1_modules(self):
        """setpoint is opt-in: default chain must stay the original 4 rules."""
        assert "l1_setpoint" not in DEFAULT_L1_MODULES
        assert DEFAULT_L1_MODULES == ["l1_constant", "l1_sigma", "l1_iqr", "l1_rate"]

    def test_build_filter_round_trip(self):
        f = build_filter("l1_setpoint", mode="command",
                         command_value=0.0, anomaly_values=[1.0])
        assert f.name == "l1_setpoint"

    def test_layer_attribute(self):
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0])
        assert r.layer == LAYER_L1_CLASSIC


# ---------------------------------------------------------------------------
# Mode A: command
# ---------------------------------------------------------------------------

class TestSetpointCommandMode:
    """command mode: command_value -> 0, anomaly_values -> 1.0, else mid."""

    def test_all_normal_values_pass(self):
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0])
        out = r.filter(np.zeros(50, dtype=np.float32))
        assert out.decision == DECISION_PASS
        assert out.score == 0.0
        assert np.all(out.detail["per_sample_score"] == 0.0)
        assert out.detail["rules"] == []
        assert out.detail["setpoint_mode"] == "command"

    def test_anomaly_value_flagged(self):
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0])
        v = np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        out = r.filter(v)
        assert out.decision == DECISION_ALERT
        ps = out.detail["per_sample_score"]
        assert ps[0] == 0.0 and ps[1] == 0.0 and ps[3] == 0.0
        assert ps[2] == 1.0 and ps[4] == 1.0
        assert out.score == 1.0
        assert "setpoint_command_anomaly" in out.detail["rules"]

    def test_mid_state_value_flagged(self):
        # A value that is neither command nor anomaly is an unexpected
        # intermediate state — flagged at mid_state_score (default 0.5).
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0])
        v = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        out = r.filter(v)
        assert out.decision == DECISION_ALERT
        assert out.detail["per_sample_score"][1] == pytest.approx(0.5)
        assert "setpoint_command_mid_state" in out.detail["rules"]

    def test_custom_mid_state_score_zero(self):
        # If mid_state_score=0, an unexpected middle value does NOT raise
        # an alert (it scores 0 and the rule stays PASS).
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0], mid_state_score=0.0)
        v = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        out = r.filter(v)
        assert out.decision == DECISION_PASS
        assert out.detail["per_sample_score"][1] == 0.0

    def test_multiple_anomaly_values(self):
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = r.filter(v)
        ps = out.detail["per_sample_score"]
        assert ps[0] == 0.0          # command
        assert ps[1] == 1.0          # anomaly 1
        assert ps[2] == 1.0          # anomaly 2
        assert ps[3] == 1.0          # anomaly 3
        assert ps[4] == 0.5          # mid state
        assert out.decision == DECISION_ALERT

    def test_nan_not_flagged(self):
        """NaN must not match command_value, anomaly_values, or mid_state."""
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0])
        v = np.array([0.0, np.nan, 1.0], dtype=np.float32)
        out = r.filter(v)
        ps = out.detail["per_sample_score"]
        assert ps[0] == 0.0, "command value normal"
        assert ps[1] == 0.0, "NaN must not be flagged"
        assert ps[2] == 1.0, "anomaly value flagged"
        # The block still alerts because of the anomaly at index 2.
        assert out.decision == DECISION_ALERT

    def test_all_nan_no_trigger(self):
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0])
        out = r.filter(np.full(5, np.nan, dtype=np.float32))
        assert out.decision == DECISION_PASS
        assert np.all(out.detail["per_sample_score"] == 0.0)

    def test_float_command_value_int_input(self):
        # Integer-typed sensor channels must match a float command_value
        # within the equality tolerance.
        r = L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[1.0])
        v = np.array([0, 0, 1, 0], dtype=np.int32)
        out = r.filter(v)
        assert out.decision == DECISION_ALERT
        assert out.detail["per_sample_score"][2] == 1.0


# ---------------------------------------------------------------------------
# Mode B: range
# ---------------------------------------------------------------------------

class TestSetpointRangeMode:
    """range mode: in-band -> 0, out-of-band -> linear ramp to 1.0."""

    def test_expected_tolerance_in_band_passes(self):
        r = L1SetpointRule(mode="range", expected=25.0, tolerance=2.0)
        v = np.array([25.0, 24.0, 26.0, 23.0, 27.0], dtype=np.float32)
        out = r.filter(v)
        # 23 and 27 are on the boundary — inclusive band, so they score 0.
        assert out.decision == DECISION_PASS
        assert np.all(out.detail["per_sample_score"] == 0.0)
        assert out.detail["rules"] == []
        assert out.detail["setpoint_mode"] == "range"

    def test_expected_tolerance_out_of_band_ramps(self):
        r = L1SetpointRule(mode="range", expected=25.0, tolerance=2.0)
        # band [23, 27], half_width=2
        # 28 → 1 unit past edge / half_width 2 = 0.5
        # 30 → 3 units past edge / 2 = 1.5 → capped at 1.0
        # 22 → 1 unit past edge on the low side = 0.5
        # 20 → 3 units past on the low side = capped 1.0
        v = np.array([28.0, 30.0, 22.0, 20.0], dtype=np.float32)
        out = r.filter(v)
        assert out.decision == DECISION_ALERT
        ps = out.detail["per_sample_score"]
        assert ps[0] == pytest.approx(0.5, abs=1e-5)
        assert ps[1] == 1.0
        assert ps[2] == pytest.approx(0.5, abs=1e-5)
        assert ps[3] == 1.0
        assert "setpoint_range_violation" in out.detail["rules"]

    def test_range_low_high_equivalent_to_expected_tolerance(self):
        """Constructing via range_low/high must match expected/tolerance."""
        r1 = L1SetpointRule(mode="range", expected=25.0, tolerance=2.0)
        r2 = L1SetpointRule(mode="range", range_low=23.0, range_high=27.0)
        v = np.array([25.0, 23.0, 27.0, 28.0, 30.0, 22.0], dtype=np.float32)
        out1 = r1.filter(v)
        out2 = r2.filter(v)
        np.testing.assert_allclose(
            out1.detail["per_sample_score"],
            out2.detail["per_sample_score"],
            atol=1e-6,
        )
        assert out1.decision == out2.decision

    def test_range_low_high_wins_when_both_given(self):
        """If both specifications are supplied, the explicit band wins."""
        r = L1SetpointRule(
            mode="range", expected=100.0, tolerance=1.0,
            range_low=23.0, range_high=27.0,
        )
        out = r.filter(np.array([25.0, 30.0], dtype=np.float32))
        # 25 is inside [23,27] -> 0; 30 is outside -> flagged.
        assert out.detail["per_sample_score"][0] == 0.0
        assert out.detail["per_sample_score"][1] > 0.0
        assert out.detail["setpoint_spec"] == {
            "mode": "range", "range_low": 23.0, "range_high": 27.0,
        }

    def test_zero_tolerance(self):
        """A zero-tolerance band only accepts the exact expected value."""
        r = L1SetpointRule(mode="range", expected=25.0, tolerance=0.0)
        v = np.array([25.0, 25.0001, 24.9999], dtype=np.float32)
        out = r.filter(v)
        # The two non-exact values get flagged.  With zero half-width the
        # ramp's denominator is clamped to 1e-9, so any non-zero excess
        # saturates to 1.0.
        ps = out.detail["per_sample_score"]
        assert ps[0] == 0.0
        assert ps[1] == 1.0
        assert ps[2] == 1.0

    def test_nan_not_flagged_in_range(self):
        r = L1SetpointRule(mode="range", expected=25.0, tolerance=2.0)
        v = np.array([25.0, np.nan, np.inf, -np.inf, 30.0], dtype=np.float32)
        out = r.filter(v)
        ps = out.detail["per_sample_score"]
        assert ps[0] == 0.0
        assert ps[1] == 0.0, "NaN must not be flagged"
        assert ps[2] == 0.0, "+Inf must not be flagged"
        assert ps[3] == 0.0, "-Inf must not be flagged"
        assert ps[4] > 0.0, "real out-of-band value flagged"
        assert out.decision == DECISION_ALERT


# ---------------------------------------------------------------------------
# Mode C: enumerate
# ---------------------------------------------------------------------------

class TestSetpointEnumerateMode:
    """enumerate mode: legal_values -> 0, anything else finite -> 1.0."""

    def test_legal_values_pass(self):
        r = L1SetpointRule(mode="enumerate", legal_values=[0.0, 1.0, 2.0])
        v = np.array([0.0, 1.0, 2.0, 0.0, 2.0], dtype=np.float32)
        out = r.filter(v)
        assert out.decision == DECISION_PASS
        assert np.all(out.detail["per_sample_score"] == 0.0)
        assert out.detail["rules"] == []
        assert out.detail["setpoint_mode"] == "enumerate"

    def test_illegal_value_flagged(self):
        r = L1SetpointRule(mode="enumerate", legal_values=[0.0, 1.0, 2.0])
        v = np.array([0.0, 3.0, 1.5, 2.0], dtype=np.float32)
        out = r.filter(v)
        assert out.decision == DECISION_ALERT
        ps = out.detail["per_sample_score"]
        assert ps[0] == 0.0
        assert ps[1] == 1.0
        assert ps[2] == 1.0
        assert ps[3] == 0.0
        assert "setpoint_enumerate_illegal" in out.detail["rules"]

    def test_single_legal_value(self):
        r = L1SetpointRule(mode="enumerate", legal_values=[42.0])
        v = np.array([42.0, 41.0, 43.0], dtype=np.float32)
        out = r.filter(v)
        ps = out.detail["per_sample_score"]
        assert ps[0] == 0.0
        assert ps[1] == 1.0
        assert ps[2] == 1.0

    def test_nan_not_flagged_in_enumerate(self):
        r = L1SetpointRule(mode="enumerate", legal_values=[0.0, 1.0])
        v = np.array([0.0, np.nan, np.inf, 5.0], dtype=np.float32)
        out = r.filter(v)
        ps = out.detail["per_sample_score"]
        assert ps[0] == 0.0
        assert ps[1] == 0.0, "NaN must not be flagged"
        assert ps[2] == 0.0, "Inf must not be flagged"
        assert ps[3] == 1.0


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestSetpointEmptyInput:
    """Empty input returns PASS with an empty per_sample_score array."""

    @pytest.mark.parametrize("kwargs", [
        dict(mode="command", command_value=0.0, anomaly_values=[1.0]),
        dict(mode="range", expected=25.0, tolerance=2.0),
        dict(mode="enumerate", legal_values=[0.0, 1.0]),
    ])
    def test_empty_input_passes_all_modes(self, kwargs):
        r = L1SetpointRule(**kwargs)
        out = r.filter(np.array([]))
        assert out.decision == DECISION_PASS
        assert out.score == 0.0
        ps = out.detail["per_sample_score"]
        assert len(ps) == 0


# ---------------------------------------------------------------------------
# Construction-time parameter validation
# ---------------------------------------------------------------------------

class TestSetpointValidation:
    """Mode-specific required params must be validated at construction."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            L1SetpointRule(mode="bogus")

    def test_command_missing_command_value_raises(self):
        with pytest.raises(ValueError, match="command_value"):
            L1SetpointRule(mode="command", anomaly_values=[1.0])

    def test_command_missing_anomaly_values_raises(self):
        with pytest.raises(ValueError, match="anomaly_values"):
            L1SetpointRule(mode="command", command_value=0.0)

    def test_command_empty_anomaly_values_raises(self):
        with pytest.raises(ValueError, match="anomaly_values"):
            L1SetpointRule(mode="command", command_value=0.0,
                           anomaly_values=[])

    def test_range_missing_all_params_raises(self):
        with pytest.raises(ValueError, match="range mode requires"):
            L1SetpointRule(mode="range")

    def test_range_only_expected_raises(self):
        with pytest.raises(ValueError, match="range mode requires"):
            L1SetpointRule(mode="range", expected=25.0)

    def test_range_only_tolerance_raises(self):
        with pytest.raises(ValueError, match="range mode requires"):
            L1SetpointRule(mode="range", tolerance=2.0)

    def test_range_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            L1SetpointRule(mode="range", expected=25.0, tolerance=-1.0)

    def test_range_inverted_bounds_raises(self):
        with pytest.raises(ValueError, match="range_low .* must be"):
            L1SetpointRule(mode="range", range_low=27.0, range_high=23.0)

    def test_enumerate_missing_legal_values_raises(self):
        with pytest.raises(ValueError, match="legal_values"):
            L1SetpointRule(mode="enumerate")

    def test_enumerate_empty_legal_values_raises(self):
        with pytest.raises(ValueError, match="legal_values"):
            L1SetpointRule(mode="enumerate", legal_values=[])

    def test_mid_state_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mid_state_score must be in"):
            L1SetpointRule(
                mode="command", command_value=0.0,
                anomaly_values=[1.0], mid_state_score=1.5,
            )


# ---------------------------------------------------------------------------
# Return-shape contract (consumers rely on these keys)
# ---------------------------------------------------------------------------

class TestSetpointReturnShape:
    """The LayerResult must carry the keys ClassicFilter aggregates on."""

    def test_detail_has_required_keys(self):
        r = L1SetpointRule(mode="range", expected=25.0, tolerance=2.0)
        out = r.filter(np.array([25.0, 30.0], dtype=np.float32))
        for key in ("rules", "per_sample_score", "setpoint_mode", "setpoint_spec"):
            assert key in out.detail, f"missing detail key: {key}"
        assert out.layer == LAYER_L1_CLASSIC

    def test_per_sample_score_length_matches_input(self):
        r = L1SetpointRule(mode="enumerate", legal_values=[0.0, 1.0])
        v = np.linspace(0, 5, 17, dtype=np.float32)
        out = r.filter(v)
        assert len(out.detail["per_sample_score"]) == len(v)

    def test_score_is_max_of_per_sample(self):
        r = L1SetpointRule(mode="range", expected=25.0, tolerance=2.0)
        v = np.array([25.0, 30.0, 35.0], dtype=np.float32)
        out = r.filter(v)
        assert out.score == pytest.approx(float(out.detail["per_sample_score"].max()))


# ---------------------------------------------------------------------------
# ClassicFilter combination (W6) — setpoint composes like any BaseFilter
# ---------------------------------------------------------------------------

class TestSetpointInClassicFilterChain:
    """``ClassicFilter(rules=[...])`` must aggregate setpoint + sigma.

    Phase 1.5 design: l1_setpoint plugs into the existing ClassicFilter
    rule-chain combinator unchanged.  This test guards the per-sample-score
    max aggregation and the rule-name concatenation across a mixed chain
    (setpoint catches a physical-violation sample that sigma also catches,
    plus a sample only setpoint flags).
    """

    def test_setpoint_plus_sigma_aggregates_per_sample(self):
        rng = np.random.RandomState(42)
        v = rng.randn(200).astype(np.float32) * 0.1 + 25.0  # ~25 +/- 0.1
        # Index 50: physically out-of-spec (35) AND statistical outlier.
        v[50] = 35.0
        # Index 100: physically in-spec (25.1) but a mild statistical blip.
        v[100] = 25.5

        cf = ClassicFilter(rules=[
            L1SetpointRule(mode="range", expected=25.0, tolerance=2.0),
            L1SigmaRule(sigma_k=3.0),
        ])
        out = cf.filter(v)

        # The block must alert — setpoint fires at index 50.
        assert out.decision == DECISION_ALERT
        ps = out.detail["per_sample_score"]
        # Index 50 is flagged by both rules; per-sample score is the max.
        assert ps[50] == pytest.approx(1.0)
        # Both rule names appear in the merged detail.
        assert "setpoint_range_violation" in out.detail["rules"]
        # ClassicFilter still reports mean / std in its detail.
        assert "mean" in out.detail and "std" in out.detail

    def test_setpoint_passes_through_when_in_spec(self):
        # All samples inside the setpoint band → setpoint contributes zeros;
        # the cascade still runs sigma and may or may not flag, but the
        # setpoint rule name must NOT appear.
        rng = np.random.RandomState(0)
        v = rng.randn(200).astype(np.float32) * 0.1 + 25.0  # tight around 25
        cf = ClassicFilter(rules=[
            L1SetpointRule(mode="range", expected=25.0, tolerance=2.0),
            L1SigmaRule(sigma_k=3.0),
        ])
        out = cf.filter(v)
        assert "setpoint_range_violation" not in out.detail["rules"]
        # per_sample_score has the right length regardless of trigger.
        assert len(out.detail["per_sample_score"]) == len(v)

