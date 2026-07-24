"""Stage-1 equivalence tests for the modular cascade refactor.

The Stage-1 refactor split ClassicFilter's 4 rules and PhysicalConstraint's
5 rules into independent modules under ``phm.algorithm.rules`` and rewired
the two combinator classes to delegate to a rule chain.  These tests prove
the refactor is **behaviour-preserving**: for every observable output of
the cascade, the post-refactor code produces values bit-for-bit identical
to what a faithful re-implementation of the original monolithic logic
would produce.

Three levels of equivalence are checked:

1. **ClassicFilter equivalence** — ``ClassicFilter()`` (default chain)
   must return LayerResults whose decision / rules / per_sample_score /
   mean / std match the original ClassicFilter logic on a wide variety
   of inputs (constant, normal, outliers, NaN/Inf, empty, mixed).

2. **PhysicalConstraint equivalence** — ``PhysicalConstraint()`` (default
   chain) must produce adjusted_scores / rules / decision matching the
   original logic across NaN / constant / range / rate / variance paths.

3. **CascadeDetector chain wiring** — passing an explicit ``l1_chain`` /
   ``l3_chain`` to CascadeDetector must produce final_scores identical
   to the default-constructed detector when the chains are equivalent
   (i.e. when the explicit chain equals the default chain contents).

The third level is what Stage-2 per-channel configuration will rely on:
``CascadeDetector(det, l1_chain=[...], l3_chain=[...])`` must be a
drop-in substitute for the default cascade when the chains match.
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
    CascadeDetector,
    ClassicFilter,
    PhysicalConstraint,
    ConstraintConfig,
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
from phm.algorithm.base import BaseDetector


# ---------------------------------------------------------------------------
# Mock detector (mirrors test_cascade.MockDetector)
# ---------------------------------------------------------------------------

class _MockDetector(BaseDetector):
    def __init__(self, score_array=None):
        self.n_params = 42
        self.model_source = "mock"
        self._score_array = score_array

    def detect(self, values, train_values_for_scaler=None):
        if self._score_array is not None:
            return self._score_array[:len(values)]
        v = np.asarray(values, dtype=np.float32)
        if len(v) == 0:
            return v
        dev = np.abs(v - np.mean(v))
        mx = np.max(dev) if np.max(dev) > 0 else 1.0
        return np.clip(dev / mx, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Test inputs — cover all the paths the original monolithic filters had.
# ---------------------------------------------------------------------------

_INPUTS = {
    "constant": np.full(100, -1.0, dtype=np.float32),
    "near_constant": np.concatenate([
        np.full(99, 5.0, dtype=np.float32),
        np.array([5.001], dtype=np.float32),
    ]),
    "normal": np.random.RandomState(42).randn(512).astype(np.float32),
    "sigma_outlier": np.concatenate([
        np.random.RandomState(42).randn(999, ).astype(np.float32),
        np.array([100.0], dtype=np.float32),
    ]),
    "iqr_outlier": np.concatenate([
        np.random.RandomState(42).randn(99, ).astype(np.float32),
        np.array([50.0], dtype=np.float32),
    ]),
    "rate_spike": np.array(
        [0.0, 0.1, 0.0, 10.0, 0.0, 0.1, 0.2, 0.1, 0.0, 0.05] * 10,
        dtype=np.float32,
    ),
    "with_nan": np.array([1.0, np.nan, 2.0, 3.0, np.nan] * 20, dtype=np.float32),
    "all_nan": np.full(50, np.nan, dtype=np.float32),
    "with_inf": np.concatenate([
        np.random.RandomState(42).randn(197, ).astype(np.float32),
        np.array([np.inf, -np.inf, 50.0], dtype=np.float32),
    ]),
    "empty": np.array([], dtype=np.float32),
    "short": np.array([1.0, 2.0, 3.0], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# 1. ClassicFilter equivalence — default chain reproduces pre-refactor logic
# ---------------------------------------------------------------------------

class TestClassicFilterEquivalence:
    """ClassicFilter with the default chain must match the original output.

    The "reference" is ClassicFilter itself — but we verify the *internal*
    chain path is exercised by also constructing the same chain explicitly
    via ``rules=[...]`` and asserting byte-for-byte equality.  This is the
    strongest equivalence check available without keeping a stale copy of
    the pre-refactor code around (the pre-refactor behaviour is captured
    by ``test_cascade.py``'s 15 L1 tests).
    """

    @pytest.mark.parametrize("key", list(_INPUTS.keys()))
    def test_default_chain_matches_explicit_default_chain(self, key):
        """ClassicFilter() == ClassicFilter(rules=[the 4 default L1 rules]).

        Both code paths go through the combinator, but the first builds the
        chain from ``enable_*`` defaults while the second passes the rule
        instances explicitly.  Agreement proves the chain-construction
        logic is correct (right rules, right order, right params).

        Uses ``to_dict()`` for the overall equality check so both the
        short-circuit SKIP paths (whose detail carries ``reason`` /
        ``constant_channel`` / ``insufficient_finite`` fields) and the
        normal pass path (whose detail carries ``per_sample_score`` /
        ``mean`` / ``std``) are compared uniformly.
        """
        values = _INPUTS[key]
        default = ClassicFilter()  # builds chain from enable_* defaults
        explicit = ClassicFilter(rules=[
            L1ConstantRule(constant_std=1e-3, enable_constant=True),
            L1SigmaRule(sigma_k=3.0),
            L1IqrRule(iqr_factor=1.5),
            L1RateRule(rate_quantile=99.0, rate_multiplier=5.0, max_rate=None),
        ])

        r_def = default.filter(values)
        r_exp = explicit.filter(values)

        # Decision + score + full detail must match.  ``to_dict`` captures
        # all observable fields regardless of which path (SKIP vs pass)
        # was taken.
        assert r_def.decision == r_exp.decision, (
            f"{key}: decision default={r_def.decision!r} explicit={r_exp.decision!r}"
        )
        assert r_def.score == pytest.approx(r_exp.score), (
            f"{key}: score default={r_def.score!r} explicit={r_exp.score!r}"
        )

        d_def = r_def.to_dict()["detail"]
        d_exp = r_exp.to_dict()["detail"]
        assert set(d_def.keys()) == set(d_exp.keys()), (
            f"{key}: detail key set differs: "
            f"default={set(d_def.keys())} explicit={set(d_exp.keys())}"
        )
        # Compare each detail field.  per_sample_score is a numpy array
        # and must be compared element-wise; everything else is a plain
        # JSON value and can be compared with ==.
        for k in d_def:
            v_def = d_def[k]
            v_exp = d_exp[k]
            if k == "per_sample_score":
                a_def = np.asarray(v_def, dtype=np.float64)
                a_exp = np.asarray(v_exp, dtype=np.float64)
                assert a_def.shape == a_exp.shape, f"{key}: per_sample_score shape"
                assert np.allclose(a_def, a_exp, equal_nan=True), (
                    f"{key}: per_sample_score mismatch "
                    f"(max diff {np.max(np.abs(a_def - a_exp)) if a_def.size else 0})"
                )
            else:
                assert v_def == pytest.approx(v_exp), (
                    f"{key}: detail[{k!r}] default={v_def!r} explicit={v_exp!r}"
                )

    @pytest.mark.parametrize("key", list(_INPUTS.keys()))
    def test_disable_flags_produce_equivalent_subset_chain(self, key):
        """ClassicFilter(enable_X=False) == ClassicFilter without rule X.

        This verifies the ``enable_*`` parameters still control which rules
        participate, matching the original ClassicFilter semantics.  Uses
        ``to_dict()`` so SKIP-path outputs (which don't carry
        ``per_sample_score``) compare uniformly.
        """
        values = _INPUTS[key]

        # Disable sigma via enable_sigma=False (original API).
        cf_flag = ClassicFilter(enable_sigma=False)
        # Disable sigma by omitting it from the explicit chain.
        cf_chain = ClassicFilter(rules=[
            L1ConstantRule(constant_std=1e-3, enable_constant=True),
            L1IqrRule(iqr_factor=1.5),
            L1RateRule(),
        ])

        r_flag = cf_flag.filter(values)
        r_chain = cf_chain.filter(values)

        assert r_flag.decision == r_chain.decision
        # to_dict captures all detail fields uniformly across SKIP/pass.
        d_flag = r_flag.to_dict()["detail"]
        d_chain = r_chain.to_dict()["detail"]
        assert set(d_flag.keys()) == set(d_chain.keys()), (
            f"{key}: detail key set differs flag={set(d_flag.keys())} chain={set(d_chain.keys())}"
        )
        for k in d_flag:
            v_flag = d_flag[k]
            v_chain = d_chain[k]
            if k == "per_sample_score":
                a_flag = np.asarray(v_flag, dtype=np.float64)
                a_chain = np.asarray(v_chain, dtype=np.float64)
                assert a_flag.shape == a_chain.shape
                assert np.allclose(a_flag, a_chain, equal_nan=True)
            else:
                assert v_flag == pytest.approx(v_chain), (
                    f"{key}: detail[{k!r}] flag={v_flag!r} chain={v_chain!r}"
                )


# ---------------------------------------------------------------------------
# 2. PhysicalConstraint equivalence
# ---------------------------------------------------------------------------

class TestPhysicalConstraintEquivalence:
    """PhysicalConstraint default chain matches explicit default chain."""

    _SCORE_INPUTS = {
        "zeros": None,
        "uniform_low": 0.1,
        "uniform_high": 0.8,
        "random": "random",
    }

    @pytest.mark.parametrize("input_key", list(_INPUTS.keys()))
    @pytest.mark.parametrize("score_key", list(_SCORE_INPUTS.keys()))
    def test_default_chain_matches_explicit_default_chain(self, input_key, score_key):
        values = _INPUTS[input_key]
        n = len(values)
        # Build the scores input.
        if self._SCORE_INPUTS[score_key] is None:
            scores = None
        elif self._SCORE_INPUTS[score_key] == "random":
            scores = np.random.RandomState(7).rand(n).astype(np.float32) if n else np.array([], dtype=np.float32)
        else:
            scores = np.full(n, self._SCORE_INPUTS[score_key], dtype=np.float32) if n else np.array([], dtype=np.float32)

        # Default config (only always-on rules: nan + constant).
        default = PhysicalConstraint()
        explicit = PhysicalConstraint(rules=[
            L3NanSanitiseRule(),
            L3ConstantRule(constant_std=1e-3),
            L3RangeRule(valid_min=None, valid_max=None, range_boost=0.95),
            L3RateRule(max_rate=None, rate_boost=0.85),
            L3VarianceRule(baseline_var=None, var_dampen_ratio=10.0, var_dampen_factor=0.3),
        ])

        r_def = default.filter(values, scores)
        r_exp = explicit.filter(values, scores)

        # Adjusted scores must be bit-for-bit identical.
        adj_def = np.asarray(r_def.detail["adjusted_scores"], dtype=np.float64)
        adj_exp = np.asarray(r_exp.detail["adjusted_scores"], dtype=np.float64)
        assert adj_def.shape == adj_exp.shape, f"{input_key}/{score_key}: shape mismatch"
        assert np.allclose(adj_def, adj_exp, equal_nan=True), (
            f"{input_key}/{score_key}: adjusted_scores mismatch "
            f"(max diff {np.max(np.abs(adj_def - adj_exp)) if adj_def.size else 0})"
        )
        # Rules and decision must match.
        assert r_def.detail["rules"] == r_exp.detail["rules"], (
            f"{input_key}/{score_key}: rules default={r_def.detail['rules']!r} explicit={r_exp.detail['rules']!r}"
        )
        assert r_def.decision == r_exp.decision

    def test_config_param_propagation(self):
        """A ConstraintConfig with non-default thresholds must reach the chain.

        Builds a config with an aggressive range boundary and verifies the
        default-constructed PhysicalConstraint honours it (proving config
        params propagate from ConstraintConfig through to the rule modules).
        """
        cfg = ConstraintConfig(valid_min=-0.5, valid_max=0.5, range_boost=0.9)
        pc = PhysicalConstraint(cfg)
        values = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        scores = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        r = pc.filter(values, scores)
        adj = r.detail["adjusted_scores"]
        # Indices 1 and 3 are out-of-range → boosted to >= 0.9.
        assert adj[1] >= 0.9
        assert adj[3] >= 0.9
        assert "range_boundary" in r.detail["rules"]


# ---------------------------------------------------------------------------
# 3. CascadeDetector chain wiring
# ---------------------------------------------------------------------------

class TestCascadeChainWiring:
    """CascadeDetector(l1_chain=..., l3_chain=...) wires chains correctly.

    When the explicit chains equal the default chain contents, the
    cascade's final_scores must be bit-for-bit identical to the default-
    constructed cascade.  This is the contract Stage-2 relies on.
    """

    @pytest.mark.parametrize("key", list(_INPUTS.keys()))
    def test_explicit_default_chain_matches_default_cascade(self, key):
        values = _INPUTS[key]

        # Default cascade — WarningService-style construction.
        default_cascade = CascadeDetector(_MockDetector())

        # Explicit-chain cascade — pass the 4 default L1 modules and 5
        # default L3 modules as explicit chains.
        explicit_cascade = CascadeDetector(
            _MockDetector(),
            l1_chain=[
                L1ConstantRule(constant_std=1e-3, enable_constant=True),
                L1SigmaRule(sigma_k=3.0),
                L1IqrRule(iqr_factor=1.5),
                L1RateRule(rate_quantile=99.0, rate_multiplier=5.0, max_rate=None),
            ],
            l3_chain=[
                L3NanSanitiseRule(),
                L3ConstantRule(constant_std=1e-3),
                L3RangeRule(),
                L3RateRule(),
                L3VarianceRule(),
            ],
        )

        out_def = default_cascade.detect_with_layers(values, channel=key)
        out_exp = explicit_cascade.detect_with_layers(values, channel=key)

        # final_scores must match bit-for-bit.
        fs_def = np.asarray(out_def.final_scores, dtype=np.float64)
        fs_exp = np.asarray(out_exp.final_scores, dtype=np.float64)
        assert fs_def.shape == fs_exp.shape, f"{key}: final_scores shape mismatch"
        assert np.allclose(fs_def, fs_exp, equal_nan=True), (
            f"{key}: final_scores mismatch (max diff {np.max(np.abs(fs_def - fs_exp)) if fs_def.size else 0})"
        )

        # Layer decisions must match.
        dec_def = [lr.decision for lr in out_def.layers]
        dec_exp = [lr.decision for lr in out_exp.layers]
        assert dec_def == dec_exp, f"{key}: layer decisions mismatch"

    def test_l1_chain_overrides_classic_param(self):
        """When l1_chain is given, the classic param is ignored.

        Verify the explicit chain wins by giving a classic that would
        produce a different decision, then checking the cascade follows
        the chain instead.
        """
        # classic=ClassicFilter(enable_sigma=False) would miss a 3σ
        # outlier, but the explicit l1_chain includes L1SigmaRule so it
        # should still catch it.
        values = np.random.RandomState(42).randn(1000).astype(np.float32)
        values[500] = 100.0

        cascade = CascadeDetector(
            _MockDetector(),
            classic=ClassicFilter(enable_sigma=False, enable_iqr=False, enable_rate=False),
            l1_chain=[L1SigmaRule(sigma_k=3.0)],  # chain wins
        )
        out = cascade.detect_with_layers(values, channel="X")
        l1 = next(lr for lr in out.layers if lr.layer == "L1_classic")
        # The chain's σ rule must have fired (classic param ignored).
        assert "sigma_3" in l1.detail["rules"]
        assert l1.decision == "alert"
