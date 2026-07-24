"""Layer 1 — Classic statistical filter (fast pre-screening).

Goal: low-cost screening that catches obviously-anomalous or obviously-broken
data *before* the expensive DL detector runs.  Four independent checks:

1. **Constant-channel detection** — ``std < ε`` ⇒ the channel carries no
   information (e.g. TSB-UAD channels C-2/D-14/M-6/S-2/T-5 whose analog
   dimension is pinned by a command bit).  Returning ``skip`` tells the
   cascade to force score=0 and avoid wasting a TSPulse forward pass.

2. **3σ rule** — any sample beyond ``mean ± k·std`` is flagged ``alert``.

3. **IQR rule** — any sample outside ``[Q1−1.5·IQR, Q3+1.5·IQR]``.

4. **Rate-of-change limit** — consecutive-sample jump exceeds a threshold.

Stage-1 refactor: the four checks are now independent modules under
``phm.algorithm.rules`` and this class is a *combinator* that builds a
default rule chain from the constructor parameters and aggregates their
outputs.  The constructor signature and ``filter()`` return shape are
byte-for-byte backward compatible (verified by
``tests/test_rules_equivalence.py``) — existing callers (WarningService,
test_cascade.py) need no changes.

A new optional ``rules`` parameter accepts an explicit ``list[BaseFilter]``
to override the default chain (Stage-2 per-channel configuration will use
this).  When ``rules=None`` the chain is built from ``enable_*`` and the
threshold parameters, reproducing the pre-refactor behaviour exactly.
"""

from __future__ import annotations

import numpy as np

from .base_filter import BaseFilter
from .cascade_types import (
    LayerResult,
    LAYER_L1_CLASSIC,
    DECISION_PASS,
    DECISION_SKIP,
)
from .rules import (
    L1ConstantRule,
    L1SigmaRule,
    L1IqrRule,
    L1RateRule,
)

__all__ = ["ClassicFilter"]


class ClassicFilter(BaseFilter):
    """Layer-1 statistical pre-filter (rule-chain combinator).

    Args:
        enable_constant:   detect near-constant channels.
        enable_sigma:      3σ outlier check.
        enable_iqr:        IQR outlier check.
        enable_rate:       rate-of-change check.
        constant_std:      channels with std below this are treated as constant.
        sigma_k:           number of standard deviations for the σ rule.
        iqr_factor:        multiplier for the IQR fence (classic 1.5).
        rate_quantile:     if ``max_rate`` is None it is derived from the
                           data as this quantile of absolute diffs (default
                           p99 → only extreme jumps trigger).
        rate_multiplier:   ``max_rate = p99 * multiplier``.
        max_rate:          explicit rate-of-change threshold.  Overrides the
                           quantile derivation when not None.
        rules:             explicit rule chain.  When None (default) the
                           chain is built from the parameters above so the
                           filter reproduces the pre-refactor behaviour.
                           When a list is given it overrides the defaults;
                           the ``enable_*`` / threshold params are still
                           stored as attributes for introspection but do
                           not affect the chain.
    """

    name = "classic_filter"

    def __init__(
        self,
        *,
        enable_constant: bool = True,
        enable_sigma: bool = True,
        enable_iqr: bool = True,
        enable_rate: bool = True,
        constant_std: float = 1e-3,
        sigma_k: float = 3.0,
        iqr_factor: float = 1.5,
        rate_quantile: float = 99.0,
        rate_multiplier: float = 5.0,
        max_rate: float | None = None,
        rules: list[BaseFilter] | None = None,
    ) -> None:
        # Store all original parameters as attributes (backward compat —
        # external code may introspect them).
        self.enable_constant = enable_constant
        self.enable_sigma = enable_sigma
        self.enable_iqr = enable_iqr
        self.enable_rate = enable_rate
        self.constant_std = constant_std
        self.sigma_k = sigma_k
        self.iqr_factor = iqr_factor
        self.rate_quantile = rate_quantile
        self.rate_multiplier = rate_multiplier
        self.max_rate = max_rate

        if rules is not None:
            # Explicit override — caller takes responsibility for the chain
            # contents (e.g. Stage-2 per-channel config).  We do not inject
            # the enable_* / threshold params here.
            self._chain: list[BaseFilter] = list(rules)
            self._rules_explicit = True
        else:
            self._chain = self._build_default_chain()
            self._rules_explicit = False

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def _build_default_chain(self) -> list[BaseFilter]:
        """Build the default L1 rule chain from constructor parameters.

        Order matters: the constant/empty/insufficient_finite guard runs
        first (it owns the short-circuit SKIP decisions), then σ / IQR /
        rate in descending severity order so the combinator's
        "first-triggered-rule wins" decision aggregation reproduces the
        original severity ordering (σ/IQR → alert, rate → suspicious).
        """
        chain: list[BaseFilter] = [
            # Guard rule is always present — it owns empty-input and
            # insufficient-finite checks that are NOT gated by
            # enable_constant in the original code.
            L1ConstantRule(
                constant_std=self.constant_std,
                enable_constant=self.enable_constant,
            ),
        ]
        if self.enable_sigma:
            chain.append(L1SigmaRule(sigma_k=self.sigma_k))
        if self.enable_iqr:
            chain.append(L1IqrRule(iqr_factor=self.iqr_factor))
        if self.enable_rate:
            chain.append(L1RateRule(
                rate_quantile=self.rate_quantile,
                rate_multiplier=self.rate_multiplier,
                max_rate=self.max_rate,
            ))
        return chain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """Evaluate all enabled checks on *values*.

        Returns a single :class:`LayerResult`.  The ``decision`` is the most
        severe finding (skip > alert > suspicious > pass).  ``detail`` lists
        every rule that triggered.
        """
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)

        # --- Phase 1: data-quality guard (empty / constant / insufficient) -
        # The first rule in the default chain owns the SKIP short-circuits.
        # In explicit-chain mode the first rule (whatever it is) runs first;
        # if it returns SKIP we propagate that result unchanged.
        if self._chain:
            first = self._chain[0].filter(v)
            if first.decision == DECISION_SKIP:
                return first
        else:
            first = None

        # --- Phase 2: statistical baseline (mu/sigma computed up front, --
        # matching the original code which calculated them before any    --
        # outlier rule ran).  These always appear in the returned detail. -
        finite_mask = np.isfinite(v)
        n_finite = int(finite_mask.sum())
        if n_finite >= 2:
            clean = v[finite_mask]
            mu = float(np.mean(clean))
            sigma = float(np.std(clean))
        else:
            # Should not happen in default-chain mode (the guard would have
            # returned insufficient_finite SKIP above), but stay safe in
            # explicit-chain mode.
            mu = 0.0
            sigma = 0.0

        # --- Phase 3: aggregate remaining rules ---------------------------
        per_sample = np.zeros(n, dtype=np.float32)
        rules: list[str] = []
        decision = DECISION_PASS

        # Seed with the guard rule's per-sample contribution (zero in
        # default mode, but an explicit chain's first rule may contribute).
        if first is not None:
            ps0 = first.detail.get("per_sample_score")
            if ps0 is not None:
                ps0 = np.asarray(ps0, dtype=np.float32)
                if len(ps0) == n:
                    per_sample = np.maximum(per_sample, ps0)
            rules.extend(first.detail.get("rules", []))
            if decision == DECISION_PASS and first.decision != DECISION_PASS:
                decision = first.decision

        # Run the remaining rules in chain order.  Per-sample scores are
        # merged with element-wise max; rule names are concatenated;
        # decision follows the original "first non-pass trigger wins"
        # semantics (σ/IQR → alert has priority over rate → suspicious
        # because they appear earlier in the chain).
        for rule in self._chain[1:]:
            r = rule.filter(v)
            r_ps = r.detail.get("per_sample_score")
            if r_ps is not None:
                r_ps = np.asarray(r_ps, dtype=np.float32)
                if len(r_ps) == n:
                    per_sample = np.maximum(per_sample, r_ps)
            rules.extend(r.detail.get("rules", []))
            if decision == DECISION_PASS and r.decision != DECISION_PASS:
                decision = r.decision

        rep_score = float(np.max(per_sample)) if n > 0 else 0.0
        return LayerResult(
            layer=LAYER_L1_CLASSIC,
            decision=decision,
            score=rep_score,
            detail={
                "rules": rules,
                "mean": mu,
                "std": sigma,
                "n_samples": n,
                "per_sample_score": per_sample,
            },
        )


__all__ = ["ClassicFilter"]
