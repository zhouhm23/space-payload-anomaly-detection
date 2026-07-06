"""Three-layer cascade detection — shared data types.

Defines the lightweight data structures passed between the classic filter
(Layer 1), the deep-learning detector (Layer 2) and the physical-constraint
validator (Layer 3).  Keeping them in a dedicated module avoids circular
imports between ``classic_filter``, ``cascade_detector`` and
``physical_constraint``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Layer identifiers ───────────────────────────────────────────────────────

LAYER_L1_CLASSIC = "L1_classic"
LAYER_L2_DL = "L2_dl"
LAYER_L3_PHYSICAL = "L3_physical"

# ── Decision constants ──────────────────────────────────────────────────────
# A decision determines whether the sample / window is forwarded to the
# next layer or short-circuited.
#
#   pass        — looks normal, continue to next layer
#   suspicious  — mildly unusual, continue but flag for deeper inspection
#   alert       — confidently anomalous, short-circuit (skip remaining layers)
#   skip        — data quality issue (e.g. constant channel), force score=0

DECISION_PASS = "pass"
DECISION_SUSPICIOUS = "suspicious"
DECISION_ALERT = "alert"
DECISION_SKIP = "skip"

ALL_DECISIONS = frozenset({
    DECISION_PASS, DECISION_SUSPICIOUS, DECISION_ALERT, DECISION_SKIP,
})


@dataclass
class LayerResult:
    """Outcome of a single cascade layer for one data block.

    Attributes:
        layer:   layer identifier string (L1_classic / L2_dl / L3_physical)
        decision: one of the ``DECISION_*`` constants
        score:   representative anomaly score for this layer (float, or
                 ``np.nan`` if the layer did not produce a per-point score)
        detail:  free-form diagnostic dict (e.g. rules triggered, thresholds
                 used).  Stored as JSON by SQLite so it must be JSON-serializable
                 (str / float / int / list / dict / bool / None).
    """

    layer: str
    decision: str
    score: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "decision": self.decision,
            "score": float(self.score),
            "detail": self.detail,
        }


@dataclass
class CascadeOutput:
    """Full output of the three-layer cascade for one channel block.

    Attributes:
        channel:       channel name
        final_scores:  per-sample final anomaly scores ``[T]`` (after all
                       three layers).  This is what downstream consumers
                       (WarningService, health calculation) use.
        layers:        list of LayerResult, one per layer that was actually
                       executed.  Layers short-circuited by an earlier
                       ``alert``/``skip`` are omitted.
        l1_scores:     per-sample L1 scores (``[T]``) or None if L1 did not
                       produce per-sample scores (e.g. produced a block-level
                       decision only).
        l2_scores:     per-sample L2 (TSPulse) raw scores (``[T]``) or None.
        l3_scores:     per-sample L3 adjusted scores (``[T]``) or None.
    """

    channel: str
    final_scores: np.ndarray
    layers: list[LayerResult] = field(default_factory=list)
    l1_scores: np.ndarray | None = None
    l2_scores: np.ndarray | None = None
    l3_scores: np.ndarray | None = None

    def to_dict(self, max_detail: bool = False) -> dict[str, Any]:
        """Serialise for API / SQLite.

        ``max_detail=True`` includes per-layer score arrays (heavier).
        """
        out: dict[str, Any] = {
            "channel": self.channel,
            "layers": [lr.to_dict() for lr in self.layers],
        }
        if max_detail:
            for name, arr in [
                ("l1_scores", self.l1_scores),
                ("l2_scores", self.l2_scores),
                ("l3_scores", self.l3_scores),
                ("final_scores", self.final_scores),
            ]:
                if arr is not None:
                    out[name] = np.asarray(arr, dtype=float).tolist()
        return out


__all__ = [
    "LAYER_L1_CLASSIC",
    "LAYER_L2_DL",
    "LAYER_L3_PHYSICAL",
    "DECISION_PASS",
    "DECISION_SUSPICIOUS",
    "DECISION_ALERT",
    "DECISION_SKIP",
    "ALL_DECISIONS",
    "LayerResult",
    "CascadeOutput",
]
