"""Unit tests for the persistence post-filter (Layer 3.5).

Covers:
  - apply_persistence: pure function semantics, edge cases, prefix-sum
    correctness, invalid W/K.
  - knee_threshold: monotonicity, all-zero fallback, basic elbow detection.
  - PersistenceFilter: streaming across blocks, per-channel isolation,
    reset semantics, warm-up behaviour.
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

from phm.algorithm.persistence_filter import (
    DEFAULT_PERSIST_K,
    DEFAULT_PERSIST_W,
    PersistenceConfig,
    PersistenceFilter,
    apply_persistence,
    knee_threshold,
    causal_ema,
)


# ---------------------------------------------------------------------------
# apply_persistence
# ---------------------------------------------------------------------------

class TestApplyPersistence:
    def test_all_zeros_stays_zero(self):
        out = apply_persistence(np.zeros(20, dtype=int), W=8, K=4)
        assert out.sum() == 0

    def test_all_ones_stays_one(self):
        # 20 ones, W=8 K=4: from index 3 onwards every window of 8 has
        # >=4 ones (in fact all 8). The first 3 samples have shorter
        # trailing windows (3, 4, 5... samples) but already satisfy K=4
        # at index 3 (window 0..3 = 4 ones). So out = [0,0,0,1,1,...,1].
        out = apply_persistence(np.ones(20, dtype=int), W=8, K=4)
        assert out[:3].sum() == 0  # warm-up (<K samples available)
        assert out[3:].sum() == 17

    def test_single_spike_suppressed(self):
        # one sample in 20 — should be fully suppressed by W=8 K=4
        preds = np.zeros(20, dtype=int)
        preds[10] = 1
        out = apply_persistence(preds, W=8, K=4)
        assert out.sum() == 0

    def test_short_run_below_K_suppressed(self):
        # 3 consecutive then gap — K=4 requires >=4 in window of 8
        preds = np.zeros(20, dtype=int)
        preds[5:8] = 1  # 3 in a row
        out = apply_persistence(preds, W=8, K=4)
        assert out.sum() == 0

    def test_long_run_confirmed(self):
        # 6 consecutive ones at indices 5..10, W=8 K=4.
        # Confirmation begins once the trailing window of 8 contains >=4
        # ones. From i=8 the window 1..8 contains the 4 ones at 5,6,7,8.
        # Confirmation continues until the window slides past the run:
        # at i=12 the window 5..12 still has indices 5..10 (6 ones) → still
        # confirmed; at i=14 window 7..14 has only 8,9,10 (3 ones) → stops.
        preds = np.zeros(20, dtype=int)
        preds[5:11] = 1
        out = apply_persistence(preds, W=8, K=4)
        # Warm-up: indices 5..7 have <4 ones in trailing window
        assert out[5:8].sum() == 0
        # Confirmed: indices 8..13 (window still contains >=4 run members)
        # i=8: window 1..8 → indices 5,6,7,8 = 4 ✓
        # i=9..11: still >=4 ✓
        # i=12: window 5..12 → indices 5,6,7,8,9,10 = 6 ✓
        # i=13: window 6..13 → indices 6,7,8,9,10 = 5 ✓
        # i=14: window 7..14 → indices 7,8,9,10 = 4 ✓
        # i=15: window 8..15 → indices 8,9,10 = 3 ✗
        assert out[8:15].sum() == 7
        assert out[15:].sum() == 0

    def test_warmup_short_block(self):
        # Block shorter than W: still works, K required within available
        preds = np.array([1, 1, 1, 1], dtype=int)
        out = apply_persistence(preds, W=8, K=4)
        # All 4 samples form a complete window at i=3
        assert out[3] == 1
        assert out[:3].sum() == 0

    def test_prefix_sum_matches_brute_force(self):
        rng = np.random.default_rng(42)
        preds = (rng.random(200) > 0.5).astype(int)
        W, K = 8, 4
        out_fast = apply_persistence(preds, W=W, K=K)
        # Brute force
        out_bf = np.zeros(200, dtype=int)
        for i in range(200):
            lo = max(0, i - W + 1)
            if preds[lo:i + 1].sum() >= K:
                out_bf[i] = 1
        assert np.array_equal(out_fast, out_bf)

    def test_invalid_W_returns_input(self):
        preds = np.array([1, 0, 1, 0], dtype=int)
        out = apply_persistence(preds, W=0, K=1)
        assert np.array_equal(out, preds.astype(np.int8))

    def test_invalid_K_returns_input(self):
        preds = np.array([1, 0, 1, 0], dtype=int)
        out = apply_persistence(preds, W=4, K=5)  # K > W
        assert np.array_equal(out, preds.astype(np.int8))

    def test_empty_input(self):
        out = apply_persistence(np.array([], dtype=int), W=8, K=4)
        assert len(out) == 0

    def test_K1_dilates_ones(self):
        # K=1, W=8: any window containing >=1 one is confirmed. This
        # *dilates* every 1 into a run of W (causal). A single spike at
        # index 5 confirms indices 5..12 (window of 8 ending at each).
        preds = np.zeros(20, dtype=int)
        preds[5] = 1
        out = apply_persistence(preds, W=8, K=1)
        # Index 5: window 0..5 contains the 1 → confirm
        # Index 12: window 5..12 contains the 1 → confirm (last)
        # Index 13: window 6..13 does NOT contain index 5 → 0
        assert out[5] == 1
        assert out[12] == 1
        assert out[13] == 0
        assert out.sum() == 8  # indices 5..12

    def test_K_equals_W_requires_full_window(self):
        # Only confirm when the entire W window is positive
        preds = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=int)
        out = apply_persistence(preds, W=4, K=4)
        # First full window at i=3 (indices 0..3 all 1) → confirm
        assert out[3] == 1
        # i=4: window 1..4 all 1 → confirm
        assert out[4] == 1
        # i=5: window 2..5 has a 0 → no
        assert out[5] == 0


# ---------------------------------------------------------------------------
# knee_threshold
# ---------------------------------------------------------------------------

class TestKneeThreshold:
    def test_all_zero_returns_eps(self):
        scores = np.zeros(100)
        thr = knee_threshold(scores, eps=1e-6)
        assert thr == pytest.approx(1e-6)

    def test_few_nonzero_returns_eps(self):
        # Less than 10 non-zero samples → fallback
        scores = np.zeros(100)
        scores[:5] = 0.5
        thr = knee_threshold(scores, eps=1e-6)
        assert thr == pytest.approx(1e-6)

    def test_bimodal_picks_low_cluster_tail(self):
        # 90% near 0.1, 10% near 0.9 → knee should land near the boundary
        scores = np.concatenate([
            np.full(90, 0.1),
            np.full(10, 0.9),
        ])
        thr = knee_threshold(scores, eps=1e-6)
        # Should be > 0.1 (bulk) and <= 0.9 (tail) — the elbow sits at the
        # jump from bulk to tail.
        assert 0.1 < thr <= 0.9

    def test_monotonic_in_tail_height(self):
        # Raising the tail while keeping the bulk fixed should not lower
        # the threshold dramatically (the knee position depends on shape,
        # but it must remain above the bulk).
        bulk = np.full(80, 0.1)
        for tail_val in [0.5, 0.7, 0.9]:
            scores = np.concatenate([bulk, np.full(20, tail_val)])
            thr = knee_threshold(scores)
            assert thr > 0.1  # always above the bulk

    def test_result_ge_eps(self):
        rng = np.random.default_rng(123)
        scores = rng.random(500)
        thr = knee_threshold(scores, eps=1e-6)
        assert thr >= 1e-6


# ---------------------------------------------------------------------------
# causal_ema
# ---------------------------------------------------------------------------

class TestCausalEma:
    def test_alpha_1_is_identity(self):
        x = np.array([0.1, 0.5, 0.9, 0.2, 0.7])
        out = causal_ema(x, alpha=1.0)
        assert np.allclose(out, x)

    def test_alpha_above_1_is_identity(self):
        x = np.array([0.1, 0.5, 0.9])
        out = causal_ema(x, alpha=2.0)
        assert np.allclose(out, x)

    def test_first_sample_is_alpha_times_x(self):
        # y[0] = alpha*x[0] + (1-alpha)*0 = alpha*x[0]
        x = np.array([1.0, 0.0, 0.0])
        out = causal_ema(x, alpha=0.3)
        assert out[0] == pytest.approx(0.3)

    def test_constant_input_converges_to_constant(self):
        # EMA initialised at 0 converges to a constant input geometrically:
        # after n steps the residual is (1-alpha)^n.  With alpha=0.2 and
        # n=50 the residual is 0.8^50 ≈ 1.4e-5 — essentially converged.
        x = np.full(50, 0.5)
        out = causal_ema(x, alpha=0.2)
        assert out[-1] == pytest.approx(0.5, abs=1e-4)
        # The smoothed series is monotonically approaching 0.5
        assert np.all(np.diff(out) > 0)

    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(42)
        x = rng.random(500)
        out = causal_ema(x, alpha=0.2)
        # Smoothed signal has lower variance than raw
        assert np.std(out) < np.std(x)

    def test_recursion_formula(self):
        # Manually verify y[i] = alpha*x[i] + (1-alpha)*y[i-1]
        x = np.array([0.4, 0.6, 0.2, 0.8])
        alpha = 0.25
        out = causal_ema(x, alpha=alpha)
        y0 = alpha * 0.4
        y1 = alpha * 0.6 + (1 - alpha) * y0
        y2 = alpha * 0.2 + (1 - alpha) * y1
        y3 = alpha * 0.8 + (1 - alpha) * y2
        assert out[0] == pytest.approx(y0)
        assert out[1] == pytest.approx(y1)
        assert out[2] == pytest.approx(y2)
        assert out[3] == pytest.approx(y3)

    def test_empty_input(self):
        out = causal_ema(np.array([]), alpha=0.2)
        assert len(out) == 0

    def test_output_length_matches(self):
        x = np.linspace(0, 1, 47)
        out = causal_ema(x, alpha=0.3)
        assert len(out) == len(x)

    def test_returns_float64(self):
        x = np.array([1, 2, 3], dtype=np.float32)
        out = causal_ema(x, alpha=0.5)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# PersistenceConfig
# ---------------------------------------------------------------------------

class TestPersistenceConfig:
    def test_defaults(self):
        cfg = PersistenceConfig()
        assert cfg.W == DEFAULT_PERSIST_W
        assert cfg.K == DEFAULT_PERSIST_K
        assert cfg.history_len >= cfg.W

    def test_invalid_W_raises(self):
        with pytest.raises(ValueError):
            PersistenceConfig(W=0)

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            PersistenceConfig(W=4, K=5)
        with pytest.raises(ValueError):
            PersistenceConfig(W=4, K=0)

    def test_history_len_clamped_to_W(self):
        cfg = PersistenceConfig(W=10, K=4, history_len=2)
        assert cfg.history_len == 10


# ---------------------------------------------------------------------------
# PersistenceFilter (stateful)
# ---------------------------------------------------------------------------

class TestPersistenceFilter:
    def test_first_block_warmup(self):
        pf = PersistenceFilter(PersistenceConfig(W=8, K=4))
        preds = np.array([1, 1, 1, 1], dtype=int)  # 4 in a row, K=4
        out = pf.update("ch1", preds)
        assert out[3] == 1
        assert out[:3].sum() == 0

    def test_streaming_continues_history(self):
        # Block 1: 3 ones (below K=4, suppressed)
        # Block 2: 3 ones (combined with history, the cross-boundary window
        # reaches K=4 → confirmed)
        pf = PersistenceFilter(PersistenceConfig(W=8, K=4))
        b1 = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
        b2 = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
        o1 = pf.update("ch", b1)
        o2 = pf.update("ch", b2)
        # In b1: max run is 3 < K → 0
        assert o1.sum() == 0
        # In b2: at index 2 of b2, the trailing window covers b1[5:] + b2[:3]
        # = [0,0,0,1,1,1] from b1's tail + [1,1,1] from b2 → but history_len=8
        # so history = b1 (8 samples). Combined: b1+b2 = 16 samples.
        # The window of 8 ending at b2[2] = b1[2:8] + b2[0:3]? No — W=8 means
        # last 8 samples: b1[3:8] (5 samples: 0,0,0,0,0) + b2[0:3] (3 samples: 1,1,1)
        # = 3 ones < 4 → still 0. OK.
        # Need a 4th one — check b2[3]? b2[3]=0. So still 0.
        assert o2.sum() == 0

    def test_streaming_confirms_across_boundary(self):
        # Block 1 ends with 3 ones, block 2 starts with 1 one → 4 in window
        pf = PersistenceFilter(PersistenceConfig(W=8, K=4))
        b1 = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
        b2 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
        o1 = pf.update("ch", b1)
        o2 = pf.update("ch", b2)
        # b1: trailing run of 3 ones at end → max window count is 3 < 4
        assert o1.sum() == 0
        # b2[0]: window = b1[1:8] (7 samples, of which 3 are 1) + b2[0] (1)
        # = 4 ones → confirmed
        assert o2[0] == 1

    def test_per_channel_isolation(self):
        pf = PersistenceFilter(PersistenceConfig(W=4, K=4))
        # Channel A accumulates history; channel B starts fresh
        pf.update("A", np.array([1, 1, 1, 1], dtype=int))
        out_b = pf.update("B", np.array([1, 0, 0, 0], dtype=int))
        # B has no history — single 1 cannot reach K=4
        assert out_b.sum() == 0

    def test_reset_single_channel(self):
        pf = PersistenceFilter(PersistenceConfig(W=4, K=4))
        pf.update("A", np.array([1, 1, 1, 1], dtype=int))
        pf.update("B", np.array([1, 1, 1, 1], dtype=int))
        pf.reset("A")
        # A's next block starts fresh
        out_a = pf.update("A", np.array([1, 0, 0, 0], dtype=int))
        assert out_a.sum() == 0
        # B still has history
        out_b = pf.update("B", np.array([1, 1, 1, 1], dtype=int))
        assert out_b[3] == 1

    def test_reset_all_channels(self):
        pf = PersistenceFilter(PersistenceConfig(W=4, K=4))
        pf.update("A", np.array([1, 1, 1, 1], dtype=int))
        pf.update("B", np.array([1, 1, 1, 1], dtype=int))
        pf.reset()
        out_a = pf.update("A", np.array([1, 0, 0, 0], dtype=int))
        out_b = pf.update("B", np.array([1, 0, 0, 0], dtype=int))
        assert out_a.sum() == 0
        assert out_b.sum() == 0

    def test_history_len_truncates(self):
        cfg = PersistenceConfig(W=4, K=4, history_len=4)
        pf = PersistenceFilter(cfg)
        # Push a long block; only the last 4 samples should be retained
        pf.update("ch", np.ones(20, dtype=int))
        hist = pf._history["ch"]
        assert len(hist) == 4
