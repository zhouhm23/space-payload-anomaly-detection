"""Layer 3 — Physical-constraint validator (post-processing adjudication).

Runs *after* the DL detector (Layer 2).  Its job is to suppress false alarms
that violate physical / system priors and to boost alarms that the DL model
under-scored but which are physically impossible.

All rules are **statistical / generic** — they do not hard-code any
MSL/SMAP-specific channel knowledge.  The idea is to provide a configurable
engine that a real payload mission can parameterise with domain-specific
constraints (temperature-voltage coupling, actuator rate limits, etc.).

Five rule families (now individual modules under ``phm.algorithm.rules``):

1. **NaN / Inf sanitisation** (:class:`L3NanSanitiseRule`) — any non-finite
   input point gets score=0 so it cannot inflate downstream statistics.

2. **Constant-channel suppression** (:class:`L3ConstantRule`) — if the
   *input window* is near-constant (std < ε), force all scores to 0.
   Early-returns the chain (rules 3/4/5 are skipped), matching the
   original behaviour.

3. **Value-range boundary** (:class:`L3RangeRule`) — values outside
   ``[valid_min, valid_max]`` are physically impossible ⇒ boost toward 1.0.

4. **Rate-of-change ceiling** (:class:`L3RateRule`) — jump exceeding
   ``max_rate`` is physically unreasonable ⇒ boost score.

5. **Variance drift** (:class:`L3VarianceRule`) — window variance too far
   from baseline ⇒ dampen the score.

Stage-1 refactor: this class is now a *combinator* over a rule chain.  The
constructor signature (``ConstraintConfig``) and ``filter()`` return shape
are byte-for-byte backward compatible (verified by
``tests/test_rules_equivalence.py``).  A new optional ``rules`` parameter
accepts an explicit ``list[BaseFilter]`` for Stage-2 per-channel
configuration.
"""

from __future__ import annotations

import numpy as np

from .base_filter import BaseFilter
from .cascade_types import (
    LayerResult,
    LAYER_L3_PHYSICAL,
    DECISION_PASS,
    DECISION_ALERT,
)
from .rules import (
    L3NanSanitiseRule,
    L3ConstantRule,
    L3RangeRule,
    L3RateRule,
    L3VarianceRule,
)

__all__ = ["ConstraintConfig", "PhysicalConstraint"]


class ConstraintConfig:
    """Configuration for :class:`PhysicalConstraint`.

    All thresholds are optional (``None`` = rule disabled) so the constraint
    engine can start minimal and grow as domain knowledge is added.
    """

    def __init__(
        self,
        *,
        # Rule 1: NaN/Inf sanitisation — always on, not configurable
        # Rule 2: constant-channel suppression
        constant_std: float = 1e-3,
        # Rule 3: value-range boundary
        valid_min: float | None = None,
        valid_max: float | None = None,
        range_boost: float = 0.95,
        # Rule 4: rate-of-change ceiling
        max_rate: float | None = None,
        rate_boost: float = 0.85,
        # Rule 5: variance drift
        baseline_var: float | None = None,
        var_dampen_ratio: float = 10.0,
        var_dampen_factor: float = 0.3,
    ) -> None:
        self.constant_std = constant_std
        self.valid_min = valid_min
        self.valid_max = valid_max
        self.range_boost = range_boost
        self.max_rate = max_rate
        self.rate_boost = rate_boost
        self.baseline_var = baseline_var
        self.var_dampen_ratio = var_dampen_ratio
        self.var_dampen_factor = var_dampen_factor


class PhysicalConstraint(BaseFilter):
    """Layer-3 physical-constraint post-processor (rule-chain combinator).

    Args:
        config: a :class:`ConstraintConfig`.  If None, a default config with
                only the always-on rules (NaN sanitisation + constant
                suppression) is used.
        rules:  explicit rule chain.  When None (default) the chain is built
                from ``config`` so the filter reproduces the pre-refactor
                behaviour exactly.  When a list is given it overrides the
                defaults (Stage-2 per-channel config).
    """

    name = "physical_constraint"

    def __init__(
        self,
        config: ConstraintConfig | None = None,
        *,
        rules: list[BaseFilter] | None = None,
    ) -> None:
        self.config = config or ConstraintConfig()

        if rules is not None:
            self._chain: list[BaseFilter] = list(rules)
            self._rules_explicit = True
        else:
            self._chain = self._build_default_chain()
            self._rules_explicit = False

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def _build_default_chain(self) -> list[BaseFilter]:
        """Build the default L3 rule chain from ``self.config``.

        Order matters and matches the original ``PhysicalConstraint.filter``
        rule numbering: nan_sanitise → constant → range → rate → variance.
        The constant rule signals an early-return via ``_stop_chain`` so the
        chain stops after it triggers (rules 3/4/5 skipped), reproducing the
        original early-return.
        """
        cfg = self.config
        return [
            L3NanSanitiseRule(),
            L3ConstantRule(constant_std=cfg.constant_std),
            L3RangeRule(
                valid_min=cfg.valid_min,
                valid_max=cfg.valid_max,
                range_boost=cfg.range_boost,
            ),
            L3RateRule(max_rate=cfg.max_rate, rate_boost=cfg.rate_boost),
            L3VarianceRule(
                baseline_var=cfg.baseline_var,
                var_dampen_ratio=cfg.var_dampen_ratio,
                var_dampen_factor=cfg.var_dampen_factor,
            ),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """Apply physical constraints to *scores* given raw *values*.

        Returns a :class:`LayerResult` whose ``detail`` contains:
            ``adjusted_scores`` — the modified per-sample score array
            ``rules``           — list of rule names that triggered
            ``decision``        — ``alert`` if any boost rule fired,
                                  otherwise ``pass``
        """
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)

        # Initialise the working score array exactly as the original code
        # did (zeros if None, copy-and-pad/truncate otherwise).
        if scores is None:
            cur = np.zeros(n, dtype=np.float64)
        else:
            cur = np.asarray(scores, dtype=np.float64).ravel().copy()
            if len(cur) != n:
                if len(cur) < n:
                    cur = np.concatenate([cur, np.zeros(n - len(cur))])
                else:
                    cur = cur[:n]

        merged_rules: list[str] = []
        merged_std: float | None = None

        # Run the chain.  Each rule consumes the previous rule's
        # ``adjusted_scores`` (chain semantics — L3 scores flow through).
        # The constant rule's early-return is honoured via ``_stop_chain``.
        for rule in self._chain:
            r = rule.filter(v, cur)
            adj = r.detail.get("adjusted_scores")
            if adj is not None:
                adj = np.asarray(adj, dtype=np.float64).ravel()
                if len(adj) == n:
                    cur = adj
                # If a rule returned a mismatched length (defensive —
                # shouldn't happen), keep the previous cur.
            merged_rules.extend(r.detail.get("rules", []))
            # Preserve std on the constant-suppression early-return path
            # (original code included "std" in that branch's detail only).
            if "std" in r.detail and merged_std is None:
                merged_std = r.detail["std"]
            if r.detail.get("_stop_chain"):
                break

        # Original semantics: alert if any rule fired, else pass.
        decision = DECISION_ALERT if merged_rules else DECISION_PASS
        rep_score = float(np.max(cur)) if n > 0 else 0.0

        detail: dict = {
            "rules": merged_rules,
            "adjusted_scores": cur.astype(np.float32),
        }
        # Preserve the "std" field on the constant-suppression path so
        # tests / consumers that read it (test_cascade.test_constant_channel_
        # suppression relies on rules only, but original code included std)
        # keep working.
        if merged_std is not None:
            detail["std"] = merged_std

        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=decision,
            score=rep_score,
            detail=detail,
        )


__all__ = ["ConstraintConfig", "PhysicalConstraint"]
