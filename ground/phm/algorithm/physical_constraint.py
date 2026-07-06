"""Layer 3 — Physical-constraint validator (post-processing裁决).

Runs *after* the DL detector (Layer 2).  Its job is to suppress false alarms
that violate physical / system priors and to boost alarms that the DL model
under-scored but which are physically impossible.

All rules are **statistical / generic** — they do not hard-code any
MSL/SMAP-specific channel knowledge.  The idea is to provide a configurable
engine that a real payload mission can parameterise with domain-specific
constraints (temperature-voltage coupling, actuator rate limits, etc.).

Five rule families:

1. **NaN / Inf sanitisation** — any non-finite input point gets score=0 so
   it cannot inflate downstream statistics.

2. **Constant-channel suppression** — if the *input window* is near-constant
   (std < ε), force all scores to 0.  This eliminates the ``threshold=0 →
   recall=1.0`` false positive that Day-8 analysis revealed on TSB-UAD
   constant channels (C-2/D-14/M-6/S-2/T-5).

3. **Value-range boundary** — values outside ``[valid_min, valid_max]`` are
   physically impossible ⇒ boost their score toward 1.0 regardless of what
   the DL model said.

4. **Rate-of-change ceiling** — consecutive-sample jump exceeding
   ``max_rate`` is physically unreasonable ⇒ boost score.

5. **Variance drift** — if the window's variance deviates too far from a
   reference (baseline) variance, the sensor is likely drifting rather than
   detecting a real anomaly ⇒ dampen the score.
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
    """Layer-3 physical-constraint post-processor.

    Args:
        config: a :class:`ConstraintConfig`.  If None, a default config with
                only the always-on rules (NaN sanitisation + constant
                suppression) is used.
    """

    name = "physical_constraint"

    def __init__(self, config: ConstraintConfig | None = None) -> None:
        self.config = config or ConstraintConfig()

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
        if scores is None:
            s = np.zeros(n, dtype=np.float64)
        else:
            s = np.asarray(scores, dtype=np.float64).ravel().copy()
            if len(s) != n:
                # Length mismatch — pad/truncate to match values
                if len(s) < n:
                    s = np.concatenate([s, np.zeros(n - len(s))])
                else:
                    s = s[:n]

        rules: list[str] = []
        cfg = self.config

        # --- Rule 1: NaN / Inf sanitisation -------------------------------
        non_finite = ~np.isfinite(v)
        n_nan = int(non_finite.sum())
        if n_nan > 0:
            s[non_finite] = 0.0
            rules.append("nan_inf_sanitise")

        finite_mask = np.isfinite(v)
        n_finite = int(finite_mask.sum())

        # --- Rule 2: Constant-channel suppression -------------------------
        if n_finite >= 2:
            clean = v[finite_mask]
            std = float(np.std(clean))
            if std < cfg.constant_std:
                s[:] = 0.0
                rules.append("constant_suppression")
                return LayerResult(
                    layer=LAYER_L3_PHYSICAL,
                    decision=DECISION_PASS,
                    score=0.0,
                    detail={
                        "rules": rules,
                        "adjusted_scores": s.astype(np.float32),
                        "std": std,
                    },
                )

        # --- Rule 3: Value-range boundary --------------------------------
        boosted_range = np.zeros(n, dtype=bool)
        if cfg.valid_min is not None:
            boosted_range |= finite_mask & (v < cfg.valid_min)
        if cfg.valid_max is not None:
            boosted_range |= finite_mask & (v > cfg.valid_max)
        n_range = int(boosted_range.sum())
        if n_range > 0:
            s[boosted_range] = np.maximum(s[boosted_range], cfg.range_boost)
            rules.append("range_boundary")

        # --- Rule 4: Rate-of-change ceiling -------------------------------
        if cfg.max_rate is not None and n >= 2:
            diffs = np.abs(np.diff(v))
            rate_out = np.zeros(n, dtype=bool)
            rate_out[1:] = diffs > cfg.max_rate
            n_rate = int(rate_out.sum())
            if n_rate > 0:
                s[rate_out] = np.maximum(s[rate_out], cfg.rate_boost)
                rules.append("rate_ceiling")

        # --- Rule 5: Variance drift dampening ----------------------------
        if cfg.baseline_var is not None and cfg.baseline_var > 0 and n_finite >= 4:
            clean = v[finite_mask]
            window_var = float(np.var(clean))
            ratio = window_var / cfg.baseline_var
            if ratio > cfg.var_dampen_ratio:
                # Window variance is >> baseline → likely sensor drift, not anomaly
                s *= cfg.var_dampen_factor
                rules.append("variance_drift_dampen")

        decision = DECISION_ALERT if rules else DECISION_PASS
        rep_score = float(np.max(s)) if n > 0 else 0.0
        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=decision,
            score=rep_score,
            detail={
                "rules": rules,
                "adjusted_scores": s.astype(np.float32),
            },
        )


__all__ = ["ConstraintConfig", "PhysicalConstraint"]
