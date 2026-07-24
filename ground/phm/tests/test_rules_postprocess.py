"""Unit tests for the Layer 3.5 leak-free post-processing rule modules.

Covers the three modules registered under ``DEFAULT_L35_MODULES``:
  * :class:`L3EmaSmoothingRule` — causal EMA smoothing (stateful)
  * :class:`L3PersistenceRule` — W/K temporal persistence (stateful)
  * :class:`L3KneeThresholdRule` — leak-free knee threshold (stateful)

These modules wrap the validated primitives in ``persistence_filter.py``
as :class:`BaseFilter` modules.  Unlike the L1/L3 rules (which are
stateless), the Layer 3.5 modules carry per-channel state across blocks;
the tests therefore exercise both the stateful ``filter_channel`` path
and the channel-less ``filter`` fallback.
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

from phm.algorithm import (
    DEFAULT_L35_MODULES,
    L3EmaSmoothingRule,
    L3KneeThresholdRule,
    L3PersistenceRule,
    build_filter,
)
from phm.algorithm.persistence_filter import causal_ema, knee_threshold


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

class TestLayer35Registry:
    """The three Layer 3.5 modules must be registered and chain-listed."""

    def test_default_l35_modules_order(self):
        # Order is the validated signal flow: knee → ema → persistence.
        assert DEFAULT_L35_MODULES == [
            "l3_knee_threshold", "l3_ema_smoothing", "l3_persistence",
        ]

    @pytest.mark.parametrize("name", DEFAULT_L35_MODULES)
    def test_build_filter_round_trip(self, name):
        f = build_filter(name)
        assert f.name == name

    def test_all_three_exported_from_algorithm(self):
        # The top-level package must re-export them for callers.
        from phm.algorithm import L3EmaSmoothingRule, L3KneeThresholdRule, L3PersistenceRule  # noqa: F401
        assert L3EmaSmoothingRule.name == "l3_ema_smoothing"
        assert L3KneeThresholdRule.name == "l3_knee_threshold"
        assert L3PersistenceRule.name == "l3_persistence"


# ---------------------------------------------------------------------------
# L3EmaSmoothingRule
# ---------------------------------------------------------------------------

class TestL3EmaSmoothingRule:
    """Stateful causal EMA smoothing."""

    def test_alpha_identity(self):
        """alpha=1.0 must be the identity (no smoothing, no state change)."""
        rule = L3EmaSmoothingRule(alpha=1.0)
        scores = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(4), scores)
        adj = r.detail["adjusted_scores"]
        assert np.allclose(adj, scores)
        assert r.detail["rules"] == []  # no smoothing applied
        # State must not have been seeded.
        assert "ch" not in rule._y_last

    def test_smoothing_reduces_peak(self):
        """α<1 must reduce the peak of a spike."""
        rule = L3EmaSmoothingRule(alpha=0.2)
        scores = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(5), scores)
        adj = r.detail["adjusted_scores"]
        # The spike at index 2 must be attenuated (< 1.0).
        assert adj[2] < 1.0
        # And the smoothing must leak into subsequent samples (index 3 > 0).
        assert adj[3] > 0.0
        assert "ema_smoothing" in r.detail["rules"]

    def test_cross_block_state_continuation(self):
        """Block 2's first output must resume from block 1's last y."""
        rule = L3EmaSmoothingRule(alpha=0.2)
        # Block 1: ramp up.
        s1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        r1 = rule.filter_channel("ch", np.zeros(3), s1)
        y_last = rule._y_last["ch"]
        # The stored seed must equal the last smoothed output.
        assert y_last == pytest.approx(float(r1.detail["adjusted_scores"][-1]))

        # Block 2: a single 1.0 — its smoothed value must continue from y_last,
        # not restart from 0.  If state weren't carried, block2[0] would be
        # α*1.0 = 0.2; with state it's α*1.0 + (1-α)*y_last.
        s2 = np.array([1.0], dtype=np.float32)
        r2 = rule.filter_channel("ch", np.zeros(1), s2)
        expected = 0.2 * 1.0 + 0.8 * y_last
        assert float(r2.detail["adjusted_scores"][0]) == pytest.approx(expected)

    def test_matches_causal_ema_primitive_single_block(self):
        """One block through the rule must equal causal_ema on that block."""
        rule = L3EmaSmoothingRule(alpha=0.2)
        scores = np.array([0.3, 0.7, 0.2, 0.9, 0.4], dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(5), scores)
        # Primitive starts from y=0, same as the rule's first-block seed.
        expected = causal_ema(scores.astype(np.float64), alpha=0.2)
        assert np.allclose(r.detail["adjusted_scores"], expected, atol=1e-6)

    def test_per_channel_isolation(self):
        """State for channel A must not leak into channel B."""
        rule = L3EmaSmoothingRule(alpha=0.2)
        rule.filter_channel("A", np.zeros(3), np.array([1.0, 1.0, 1.0], dtype=np.float32))
        # B's first call must start from seed 0 (not A's y_last).
        r_b = rule.filter_channel("B", np.zeros(1), np.array([1.0], dtype=np.float32))
        # α*1 + (1-α)*0 = 0.2
        assert float(r_b.detail["adjusted_scores"][0]) == pytest.approx(0.2)

    def test_reset_clears_state(self):
        rule = L3EmaSmoothingRule(alpha=0.2)
        rule.filter_channel("A", np.zeros(2), np.array([1.0, 1.0], dtype=np.float32))
        rule.filter_channel("B", np.zeros(2), np.array([1.0, 1.0], dtype=np.float32))
        rule.reset("A")
        assert "A" not in rule._y_last
        assert "B" in rule._y_last
        rule.reset()
        assert not rule._y_last

    def test_filter_without_channel_uses_default_key(self):
        """The BaseFilter.filter() entry point must work channel-less."""
        rule = L3EmaSmoothingRule(alpha=0.2)
        r = rule.filter(np.zeros(3), np.array([1.0, 1.0, 1.0], dtype=np.float32))
        assert "ema_smoothing" in r.detail["rules"]
        assert "" in rule._y_last  # default key

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            L3EmaSmoothingRule(alpha=0.0)
        with pytest.raises(ValueError):
            L3EmaSmoothingRule(alpha=1.5)

    def test_empty_block_no_state_change(self):
        rule = L3EmaSmoothingRule(alpha=0.2)
        r = rule.filter_channel("ch", np.array([]), np.array([]))
        assert r.detail["rules"] == []
        assert "ch" not in rule._y_last


# ---------------------------------------------------------------------------
# L3PersistenceRule
# ---------------------------------------------------------------------------

class TestL3PersistenceRule:
    """Stateful W/K persistence filtering."""

    def test_single_spike_suppressed(self):
        """A 1-sample spike below K must be suppressed."""
        rule = L3PersistenceRule(W=8, K=4, threshold=0.5)
        # One sample above threshold, rest below — vote can't reach K=4.
        scores = np.array([0.1, 0.1, 0.9, 0.1, 0.1], dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(5), scores)
        adj = r.detail["adjusted_scores"]
        # The spike at index 2 must be zeroed (not enough persistence).
        assert adj[2] == 0.0
        assert r.detail["n_confirmed"] == 0

    def test_sustained_anomaly_confirmed(self):
        """A run of >= K samples above threshold must be confirmed."""
        rule = L3PersistenceRule(W=8, K=4, threshold=0.5)
        # 5 consecutive samples above threshold — vote reaches K=4 by index 3.
        scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(5), scores)
        adj = r.detail["adjusted_scores"]
        # The last few samples must survive (kept their original score).
        assert adj[4] == pytest.approx(0.9)
        assert r.detail["n_confirmed"] >= 1
        assert "persistence" in r.detail["rules"]

    def test_threshold_binarisation(self):
        """The threshold controls which samples count as a positive vote."""
        rule = L3PersistenceRule(W=4, K=2, threshold=0.7)
        # Scores 0.6 are below threshold=0.7 → no positives → all suppressed.
        scores = np.full(10, 0.6, dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(10), scores)
        assert r.detail["n_confirmed"] == 0
        assert np.all(r.detail["adjusted_scores"] == 0.0)

    def test_cross_block_confirmation(self):
        """Block 1's tail + block 2's head must combine to reach K."""
        rule = L3PersistenceRule(W=8, K=4, threshold=0.5)
        # Block 1: 3 positives at the tail — not enough alone (K=4).
        s1 = np.array([0.1, 0.9, 0.9, 0.9], dtype=np.float32)
        r1 = rule.filter_channel("ch", np.zeros(4), s1)
        # Block 2: 1 more positive at the head — combined with the 3 in
        # history, the vote at block2[0] reaches 4 → confirmed.
        s2 = np.array([0.9, 0.1, 0.1], dtype=np.float32)
        r2 = rule.filter_channel("ch", np.zeros(3), s2)
        adj2 = r2.detail["adjusted_scores"]
        assert adj2[0] == pytest.approx(0.9)  # confirmed
        assert r2.detail["n_confirmed"] >= 1

    def test_per_channel_isolation(self):
        rule = L3PersistenceRule(W=4, K=2, threshold=0.5)
        # A accumulates 3 positives.
        rule.filter_channel("A", np.zeros(3), np.array([0.9, 0.9, 0.9], dtype=np.float32))
        # B's first call must not see A's history.
        r_b = rule.filter_channel("B", np.zeros(1), np.array([0.9], dtype=np.float32))
        # A single positive in a fresh channel can't reach K=2.
        assert r_b.detail["n_confirmed"] == 0

    def test_reset(self):
        rule = L3PersistenceRule(W=4, K=2, threshold=0.5)
        rule.filter_channel("A", np.zeros(3), np.array([0.9, 0.9, 0.9], dtype=np.float32))
        rule.reset("A")
        # After reset A is fresh again.
        r = rule.filter_channel("A", np.zeros(1), np.array([0.9], dtype=np.float32))
        assert r.detail["n_confirmed"] == 0

    def test_filter_without_channel(self):
        """BaseFilter.filter() entry point uses the default channel key."""
        rule = L3PersistenceRule(W=4, K=2, threshold=0.5)
        scores = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        r = rule.filter(np.zeros(4), scores)
        assert r.detail["n_confirmed"] >= 1

    def test_empty_block(self):
        rule = L3PersistenceRule()
        r = rule.filter_channel("ch", np.array([]), np.array([]))
        assert r.detail["n_confirmed"] == 0
        assert r.detail["rules"] == []

    def test_invalid_W_K_raises(self):
        with pytest.raises(ValueError):
            L3PersistenceRule(W=4, K=5)  # K > W


# ---------------------------------------------------------------------------
# L3KneeThresholdRule
# ---------------------------------------------------------------------------

class TestL3KneeThresholdRule:
    """Stateful leak-free knee threshold."""

    def test_fit_sets_threshold(self):
        """fit() must compute and cache the knee threshold."""
        rule = L3KneeThresholdRule()
        # Bimodal: bulk near 0.1, tail near 0.9 — knee sits at the elbow
        # where the tail rises above the bulk (Satopaa kneadle finds the
        # point of max distance from the baseline on the sorted curve,
        # which for a small high tail is near the tail's start ≈ 0.85).
        rng = np.random.RandomState(0)
        train = np.concatenate([
            rng.uniform(0.05, 0.15, 400),
            rng.uniform(0.85, 0.95, 100),
        ]).astype(np.float32)
        thr = rule.fit("ch", train)
        # Threshold must land above the bulk (> 0.15) and at/below the
        # tail start (≤ 0.95) — i.e. in the gap or at the elbow.
        assert 0.15 < thr <= 0.95
        assert rule.get_threshold("ch") == thr

    def test_fit_then_filter_suppresses_subthreshold(self):
        """After fit, sub-threshold samples must be zeroed."""
        rule = L3KneeThresholdRule(threshold_override=0.5)
        scores = np.array([0.1, 0.6, 0.2, 0.8, 0.3], dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(5), scores)
        adj = r.detail["adjusted_scores"]
        # Above-threshold samples keep their value.
        assert adj[1] == pytest.approx(0.6)
        assert adj[3] == pytest.approx(0.8)
        # At-or-below samples zeroed.
        assert adj[0] == 0.0
        assert adj[2] == 0.0
        assert adj[4] == 0.0
        assert r.detail["threshold"] == 0.5
        assert "knee_threshold" in r.detail["rules"]

    def test_threshold_override_skips_fit(self):
        """threshold_override must bypass accumulation entirely."""
        rule = L3KneeThresholdRule(threshold_override=0.4)
        scores = np.array([0.3, 0.5], dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(2), scores)
        adj = r.detail["adjusted_scores"]
        # Override applied immediately, no warm-up.
        assert r.detail["threshold"] == 0.4
        # 0.3 ≤ 0.4 → suppressed; 0.5 > 0.4 → kept.
        assert adj[0] == 0.0
        assert adj[1] == pytest.approx(0.5)
        # No accumulation buffer should exist.
        assert "ch" not in rule._train_buf

    def test_online_warm_up_passes_through(self):
        """Before min_fit_samples, scores pass through unchanged."""
        rule = L3KneeThresholdRule(min_fit_samples=100)
        scores = np.array([0.1, 0.6, 0.2, 0.8], dtype=np.float32)
        r = rule.filter_channel("ch", np.zeros(4), scores)
        # Warm-up: no threshold yet, scores unchanged.
        assert "threshold" not in r.detail
        assert r.detail["rules"] == []
        assert np.allclose(r.detail["adjusted_scores"], scores)
        # Buffer should have accumulated.
        assert len(rule._train_buf["ch"]) == 4

    def test_online_accumulation_derives_threshold(self):
        """Once min_fit_samples is reached, the knee is derived automatically."""
        rule = L3KneeThresholdRule(min_fit_samples=50)
        rng = np.random.RandomState(0)
        # Feed a bimodal training distribution across two blocks.
        train = np.concatenate([
            rng.uniform(0.05, 0.15, 40),
            rng.uniform(0.85, 0.95, 20),
        ]).astype(np.float32)
        # Block 1 (30 samples) — warm-up, pass through.
        r1 = rule.filter_channel("ch", np.zeros(30), train[:30])
        assert "threshold" not in r1.detail
        # Block 2 (30 samples) — crosses min_fit_samples=50, derives knee.
        r2 = rule.filter_channel("ch", np.zeros(30), train[30:])
        assert "threshold" in r2.detail
        thr = r2.detail["threshold"]
        # Knee lands in the gap or at the tail elbow.
        assert 0.15 < thr <= 0.95
        # Subsequent calls use the cached threshold.
        assert rule.get_threshold("ch") == thr

    def test_all_zero_train_falls_back_to_eps(self):
        """knee_threshold on all-zero input returns eps."""
        rule = L3KneeThresholdRule(eps=1e-5)
        thr = rule.fit("ch", np.zeros(100))
        assert thr == pytest.approx(1e-5)

    def test_matches_knee_threshold_primitive(self):
        """fit() must agree with the underlying knee_threshold primitive."""
        rule = L3KneeThresholdRule()
        rng = np.random.RandomState(42)
        train = np.concatenate([
            rng.uniform(0.0, 0.2, 300),
            rng.uniform(0.8, 1.0, 80),
        ]).astype(np.float32)
        thr_rule = rule.fit("ch", train)
        thr_prim = knee_threshold(train.astype(np.float64))
        assert thr_rule == pytest.approx(thr_prim)

    def test_per_channel_isolation(self):
        rule = L3KneeThresholdRule(threshold_override=0.5)
        rule.filter_channel("A", np.zeros(2), np.array([0.1, 0.9], dtype=np.float32))
        # B with no override-equivalent state is independent (override is
        # global, but fit/accumulation state must be per-channel).
        rule2 = L3KneeThresholdRule(min_fit_samples=10)
        rule2.filter_channel("A", np.zeros(5), np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
        r_b = rule2.filter_channel("B", np.zeros(5), np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32))
        # B is still in warm-up (5 < 10).
        assert "threshold" not in r_b.detail

    def test_reset(self):
        rule = L3KneeThresholdRule()
        rule.fit("A", np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9], dtype=np.float32))
        rule.reset("A")
        assert rule.get_threshold("A") is None

    def test_filter_without_channel(self):
        rule = L3KneeThresholdRule(threshold_override=0.5)
        r = rule.filter(np.zeros(3), np.array([0.2, 0.6, 0.7], dtype=np.float32))
        assert r.detail["threshold"] == 0.5
