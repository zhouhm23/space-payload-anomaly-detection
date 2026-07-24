"""L3 rule — NaN / Inf score sanitisation.

Ported verbatim from rule 1 of ``PhysicalConstraint.filter``
(``physical_constraint.py:132-137``).  Any non-finite *input* sample
forces its corresponding score to ``0.0`` so NaN/Inf cannot inflate
downstream statistics.  Rule name contributed to the merged detail:
``"nan_inf_sanitise"``.

Unlike most L3 rules this one is *always on* in the original code (not
gated by a config threshold) — it is a data-hygiene guard, not a
domain-knowledge constraint.  We preserve that: the rule runs unconditionally
and contributes to the score chain.
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

__all__ = ["L3NanSanitiseRule"]


@register_filter("l3_nan_sanitise")
class L3NanSanitiseRule(BaseFilter):
    """Zero out scores at non-finite input positions.

    No tunable parameters — matches the original rule which had no config.
    """

    name = "l3_nan_sanitise"

    def __init__(self) -> None:
        pass

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
                # Length mismatch — pad/truncate to match values (original
                # behaviour preserved verbatim).
                if len(s) < n:
                    s = np.concatenate([s, np.zeros(n - len(s))])
                else:
                    s = s[:n]

        rules: list[str] = []
        non_finite = ~np.isfinite(v)
        n_nan = int(non_finite.sum())
        if n_nan > 0:
            s[non_finite] = 0.0
            rules.append("nan_inf_sanitise")

        return LayerResult(
            layer=LAYER_L3_PHYSICAL,
            decision=DECISION_PASS,
            score=0.0,
            detail={
                "rules": rules,
                "adjusted_scores": s.astype(np.float32),
            },
        )
