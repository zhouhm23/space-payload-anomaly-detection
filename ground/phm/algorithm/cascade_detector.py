"""Three-layer cascade detector — orchestrates L1→L2→L3.

This is the glue that chains the classic filter (L1), the DL anomaly
detector (L2, e.g. TSPulse) and the physical-constraint validator (L3)
into a single pipeline.  Two validated enhancements are integrated:

* **L1 fusion** — the L1 classic-filter per-sample scores are blended into
  the final score via ``max(final, l1_norm * l1_fuse_weight)`` so that sharp
  statistical outliers (3σ / IQR) can lift the score even when L2
  under-reacts.  Controlled by ``l1_fuse_weight`` (default 0.3, 0 disables).

* **Per-channel calibration** — when a :class:`CalibrationConfig` is
  supplied and the channel has an entry, the L2 score gets its direction
  flipped (TSPulse reconstructs anomalous patterns well on high-anomaly
  channels) and/or the STFT frequency feature swapped in, per the offline
  LOO selection.  No config ⇒ default TSPulse-only path (backward compatible).

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
from .calibration_config import CalibrationConfig, ChannelCalibration
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
from .direction_calibrator import DirectionCalibrator
from .freq_feature import FreqFeatureExtractor
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
        l1_fuse_weight: weight for fusing L1 per-sample scores into the
                    final score (``final = max(final, l1_norm * w)``).
                    0 disables L1 fusion (legacy behaviour).  Default 0.3,
                    validated in ``experiments/cascade_eval/benchmark_cascade.py``.
        calibration_config: optional :class:`CalibrationConfig` carrying
                    per-channel offline calibration (direction flip,
                    score-type selection, freq baseline).  When None or
                    when a channel has no entry, the cascade runs in its
                    default (TSPulse-only, unflipped) mode.

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
        l1_fuse_weight: float = 0.3,
        calibration_config: CalibrationConfig | None = None,
    ) -> None:
        self._detector = detector
        self._classic = classic or ClassicFilter()
        self._constraint = constraint or PhysicalConstraint()
        self.skip_l2_on_l1_alert = skip_l2_on_l1_alert
        self.l1_fuse_weight = float(l1_fuse_weight)
        self.calibration_config = calibration_config
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
        context: np.ndarray | None = None,
    ) -> CascadeOutput:
        """Run the full three-layer cascade and return structured output.

        Args:
            context: optional preceding block prepended to the L2 detector's
                input for pipeline overlap (see AnomalyDetector.detect).
        """
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
                raw_scores = self._detector.detect(v, train_values_for_scaler, context=context)
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

            # ── Calibration: direction flip + per-channel score-type ──
            # Offline-calibrated channels may need (a) their score direction
            # inverted (TSPulse reconstructs anomalous patterns well on
            # high-anomaly-ratio channels) and/or (b) the frequency feature
            # swapped in for the TSPulse score.  Both decisions come from
            # channel_calibration.json and are no-ops when the channel has
            # no entry or no config was supplied.
            cal = (
                self.calibration_config.get(channel)
                if self.calibration_config is not None and channel
                else None
            )
            cal_detail: dict = {}
            if cal is not None:
                tsp_score = raw_scores
                # Direction flip — expects input in [0,1] (clip-normalised).
                # AnomalyDetector.detect clips to [0,1], so this holds.
                tsp_score = DirectionCalibrator.flip(tsp_score, cal.flip)
                chosen = tsp_score
                if cal.score_type in ("freq", "fusion"):
                    if cal.freq_band_mean is None or cal.freq_band_std is None:
                        logger.debug(
                            "channel %s score_type=%s but no freq baseline — "
                            "falling back to tsp",
                            channel, cal.score_type,
                        )
                    else:
                        try:
                            fe = FreqFeatureExtractor(
                                band_mean=cal.freq_band_mean,
                                band_std=cal.freq_band_std,
                                z_min=cal.freq_z_min,
                                z_max=cal.freq_z_max,
                            )
                            freq_score = fe.transform(v)
                            # Align length defensively
                            if len(freq_score) != len(tsp_score):
                                freq_score = freq_score[: len(tsp_score)]
                            if cal.score_type == "freq":
                                chosen = freq_score.astype(np.float32)
                            else:  # fusion
                                chosen = np.maximum(tsp_score, freq_score).astype(np.float32)
                        except Exception:
                            logger.warning(
                                "freq feature failed for channel %s — using tsp",
                                channel, exc_info=True,
                            )
                raw_scores = chosen
                cal_detail = {
                    "flip": cal.flip,
                    "score_type": cal.score_type,
                    "threshold": cal.threshold,
                    "threshold_name": cal.threshold_name,
                }
                l2_scores = raw_scores

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

        # ── L1 fusion ───────────────────────────────────────────────
        # Blend the L1 classic-filter per-sample scores into the final
        # score: ``final = max(final, l1_norm * l1_fuse_weight)``.  This
        # lets sharp statistical outliers (3σ / IQR) lift the final score
        # even when the DL detector under-reacts.  Disabled when
        # ``l1_fuse_weight == 0`` (legacy behaviour) or when L1 produced no
        # per-sample scores.  Validated in
        # experiments/cascade_eval/benchmark_cascade.py (L1_FUSE_WEIGHT).
        #
        # The min-max normalisation is done in pure numpy (rather than
        # fitting a new scaler on every call) — semantically identical
        # but avoids the per-call sklearn overhead in the streaming hot path.
        if self.l1_fuse_weight > 0.0 and l1_scores is not None:
            try:
                l1_arr = np.asarray(l1_scores, dtype=np.float32).ravel()
                if len(l1_arr) == len(final):
                    l1_min = float(l1_arr.min())
                    l1_max = float(l1_arr.max())
                    l1_range = l1_max - l1_min
                    if l1_range > 1e-12:
                        l1_norm = (l1_arr - l1_min) / l1_range
                    else:
                        l1_norm = np.zeros_like(l1_arr)
                    final = np.maximum(final, l1_norm * self.l1_fuse_weight).astype(
                        np.float32
                    )
            except Exception:
                logger.debug("L1 fusion failed for channel %s", channel, exc_info=True)

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
