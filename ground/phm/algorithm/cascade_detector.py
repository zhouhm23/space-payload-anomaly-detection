"""Three-layer cascade detector — orchestrates L1→L2→L3.

This is the glue that chains the classic filter (L1), the DL anomaly
detector (L2, e.g. TSPulse) and the physical-constraint validator (L3)
into a single pipeline.

Design goals:

* **Backward compatible** — ``detect()`` returns a plain ``np.ndarray`` of
  per-sample scores, exactly like :class:`BaseDetector`.  Existing callers
  (WarningService, space segment) can swap a single detector for the cascade
  without touching their code.

* **Inspectable** — ``detect_with_layers()`` returns a :class:`CascadeOutput`
  that records each layer's decision and score array.  This is what gets
  persisted to SQLite and surfaced via ``/api/detection``.

* **Short-circuiting** — if L1 returns ``skip`` (constant channel), L2 is
  never called, saving a full TSPulse forward pass.  If L1 returns ``alert``
  (obvious 3σ / IQR outlier), the per-sample L1 scores are used directly
  and L2 is skipped — the sample is already confidently flagged.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import BaseDetector
from .base_filter import BaseFilter
from .cascade_types import (
    CascadeOutput,
    LayerResult,
    LAYER_L1_CLASSIC,
    LAYER_L2_DL,
    LAYER_L3_PHYSICAL,
    DECISION_PASS,
    DECISION_ALERT,
    DECISION_SKIP,
)
from .classic_filter import ClassicFilter
from .physical_constraint import PhysicalConstraint

logger = logging.getLogger(__name__)

__all__ = ["CascadeDetector"]


class CascadeDetector(BaseDetector):
    """Three-layer cascade anomaly detector.

    Args:
        detector:   the L2 DL detector (must implement ``BaseDetector``).
        classic:    the L1 classic filter.  If None, a default
                    :class:`ClassicFilter` is created.
        constraint: the L3 physical-constraint validator.  If None, a
                    default :class:`PhysicalConstraint` is created.
        skip_l2_on_l1_alert: if True, when L1 returns ``alert`` the cascade
                    skips L2 and uses L1 per-sample scores directly.  Set
                    False to always run L2 (more thorough but slower).

    The ``n_params`` / ``model_source`` attributes are delegated to the
    wrapped L2 detector so downstream code that inspects model size still
    works.
    """

    def __init__(
        self,
        detector: BaseDetector,
        classic: BaseFilter | None = None,
        constraint: BaseFilter | None = None,
        skip_l2_on_l1_alert: bool = False,
    ) -> None:
        self._detector = detector
        self._classic = classic or ClassicFilter()
        self._constraint = constraint or PhysicalConstraint()
        self.skip_l2_on_l1_alert = skip_l2_on_l1_alert
        # Delegate model metadata
        self.n_params = getattr(detector, "n_params", 0)
        self.model_source = getattr(detector, "model_source", "cascade")

    # ------------------------------------------------------------------
    # Cascade entry point (full output)
    # ------------------------------------------------------------------

    def detect_with_layers(
        self,
        values: np.ndarray,
        train_values_for_scaler: np.ndarray | None = None,
        channel: str = "",
    ) -> CascadeOutput:
        """Run the full three-layer cascade and return structured output."""
        v = np.asarray(values, dtype=np.float32).ravel()
        n = len(v)
        layers: list[LayerResult] = []
        l1_scores: np.ndarray | None = None
        l2_scores: np.ndarray | None = None
        l3_scores: np.ndarray | None = None

        # ── Layer 1: Classic filter ──────────────────────────────────
        try:
            l1 = self._classic.filter(v)
        except Exception:
            logger.warning("L1 classic filter failed", exc_info=True)
            l1 = LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=DECISION_PASS,
                score=0.0,
                detail={"error": "l1_failed"},
            )
        layers.append(l1)
        l1_detail = l1.detail
        l1_scores = l1_detail.get("per_sample_score")
        if l1_scores is not None:
            l1_scores = np.asarray(l1_scores, dtype=np.float32)

        # Short-circuit: constant / broken channel → force zero, skip L2+L3
        if l1.decision == DECISION_SKIP:
            final = np.zeros(n, dtype=np.float32)
            # Still run L3 sanitisation (NaN etc.) but scores stay 0
            try:
                l3 = self._constraint.filter(v, final)
                layers.append(l3)
                if "adjusted_scores" in l3.detail:
                    final = l3.detail["adjusted_scores"]
                    l3_scores = final
            except Exception:
                logger.debug("L3 on skip path failed", exc_info=True)
            return CascadeOutput(
                channel=channel,
                final_scores=final,
                layers=layers,
                l1_scores=l1_scores,
                l2_scores=None,
                l3_scores=l3_scores,
            )

        # Short-circuit: obvious outlier + configured to skip L2
        if l1.decision == DECISION_ALERT and self.skip_l2_on_l1_alert and l1_scores is not None:
            raw_scores = l1_scores.copy()
        else:
            # ── Layer 2: DL detector ─────────────────────────────────
            try:
                raw_scores = self._detector.detect(v, train_values_for_scaler)
                raw_scores = np.asarray(raw_scores, dtype=np.float32).ravel()
                # Sanitise NaN/Inf from model output before L3
                nan_mask = ~np.isfinite(raw_scores)
                if nan_mask.any():
                    raw_scores[nan_mask] = 0.0
            except Exception:
                logger.warning("L2 DL detector failed", exc_info=True)
                raw_scores = np.zeros(n, dtype=np.float32)
            l2_scores = raw_scores
            layers.append(LayerResult(
                layer=LAYER_L2_DL,
                decision=DECISION_PASS,
                score=float(np.max(raw_scores)) if n > 0 else 0.0,
                detail={"model": self.model_source},
            ))

        # ── Layer 3: Physical constraint ────────────────────────────
        try:
            l3 = self._constraint.filter(v, raw_scores)
            layers.append(l3)
            if "adjusted_scores" in l3.detail:
                final = np.asarray(l3.detail["adjusted_scores"], dtype=np.float32)
            else:
                final = raw_scores.copy()
            l3_scores = final
        except Exception:
            logger.warning("L3 physical constraint failed", exc_info=True)
            final = raw_scores.copy()

        # Final safety: ensure no NaN/Inf leaks to downstream consumers
        non_finite = ~np.isfinite(final)
        if non_finite.any():
            final = final.copy()
            final[non_finite] = 0.0

        return CascadeOutput(
            channel=channel,
            final_scores=final,
            layers=layers,
            l1_scores=l1_scores,
            l2_scores=l2_scores,
            l3_scores=l3_scores,
        )

    # ------------------------------------------------------------------
    # BaseDetector-compatible interface
    # ------------------------------------------------------------------

    def detect(
        self,
        values: np.ndarray,
        train_values_for_scaler: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return only the final per-sample scores (backward-compatible)."""
        out = self.detect_with_layers(values, train_values_for_scaler)
        return out.final_scores


__all__ = ["CascadeDetector"]
