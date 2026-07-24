"""L3 rule — variance drift dampening.

Ported verbatim from rule 5 of ``PhysicalConstraint.filter``
(``physical_constraint.py:181-189``).  When the window's variance
deviates too far from a reference (baseline) variance — i.e. the
ratio ``window_var / baseline_var`` exceeds ``var_dampen_ratio`` —
the sensor is likely drifting rather than detecting a real anomaly,
so all scores are scaled by ``var_dampen_factor`` (default ``0.3``).
Rule name contributed: ``"variance_drift_dampen"``.

Disabled when ``baseline_var`` is None (the original default).
"""

from __future__ import annotations

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L3_PHYSICAL,
    DECISION_PASS,
)
from ._base import register_filter

__all__ = ["L3VarianceRule"]


# Minimum finite-sample count for variance to be meaningful — matches
# original guard ``n_finite >= 4``.
VARIANCE_MIN_FINITE = 4


@register_filter("l3_variance")
class L3VarianceRule(BaseFilter):
    """Dampen all scores when window variance drifts far from baseline.

    Args:
        baseline_var: reference (training-segment) variance.  None
            disables the rule.
        var_dampen_ratio: trigger threshold — dampen when
            ``window_var / baseline_var`` exceeds this (default ``10.0``).
        var_dampen_factor: multiplicative score scale applied on
            trigger (default ``0.3``).
    """

    name = "l3_variance"

    def __init__(
        self,
        *,
        baseline_var: float | None = None,
        var_dampen_ratio: float = 10.0,
        var_dampen_factor: float = 0.3,
    ) -> None:
        self.baseline_var = None if baseline_var is None else float(baseline_var)
        self.var_dampen_ratio = float(var_dampen_ratio)
        self.var_dampen_factor = float(var_dampen_factor)

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)
        if scores is None:
            s = np.zeros(n, dtype=np.float64)
        else:
            s = np.asarray(scores, dtype=np.float64).ravel().copy()
            if len(s) != n:
                if len(s) < n:
                    s = np.concatenate([s, np.zeros(n - len(s))])
                else:
                    s = s[:n]

        rules: list[str] = []
        if (
            self.baseline_var is not None
            and self.baseline_var > 0
        ):
            finite_mask = np.isfinite(v)
            n_finite = int(finite_mask.sum())
            if n_finite >= VARIANCE_MIN_FINITE:
                clean = v[finite_mask]
                window_var = float(np.var(clean))
                ratio = window_var / self.baseline_var
                if ratio > self.var_dampen_ratio:
                    # Window variance is >> baseline → likely sensor
                    # drift, not anomaly.
                    s *= self.var_dampen_factor
                    rules.append("variance_drift_dampen")

        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=DECISION_PASS,
            score=0.0,
            detail={
                "rules": rules,
                "adjusted_scores": s.astype(np.float32),
            },
        )
