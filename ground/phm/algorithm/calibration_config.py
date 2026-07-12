"""Per-channel calibration configuration and LOO threshold selection.

This module wires together the three offline-calibrated improvements that
were validated in ``experiments/``:

1. **Direction flip** (``DirectionCalibrator.fit``) — per-channel bool.
2. **Score-type selection** (tsp / freq / fusion) — per-channel choice,
   picked by 18-candidate leave-one-event-out.
3. **Threshold selection** — one of six candidate formulas, picked by the
   same LOO procedure.

The offline pipeline (:func:`build_calibration_for_channel`) runs all three
and produces a :class:`ChannelCalibration` dataclass, which is serialised to
``channel_calibration.json``.  At runtime :class:`CalibrationConfig` loads
that JSON and :meth:`CalibrationConfig.get` returns the per-channel record
(or ``None`` for uncalibrated channels, which then fall back to the default
TSPulse-only path).

LOO selection logic ported verbatim from
``experiments/tspulse_eval/run_per_channel_select.py:97-145`` and
``run_with_direction_calibration.py:204-289``.  Note: the experiment's LOO
loop iterates ``leave_idx`` but the body recomputes full-test FA each time
(effectively a full-test FA tiebreak, not a true leave-one-out).  This is
preserved here for numerical consistency with the published experiment
results — see the comment in :func:`loo_select_from_18`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .direction_calibrator import DirectionCalibrator
from .freq_feature import FreqFeatureExtractor

logger = logging.getLogger(__name__)

__all__ = [
    "ChannelCalibration",
    "CalibrationConfig",
    "CANDIDATE_THRESHOLDS",
    "compute_threshold",
    "find_events",
    "loo_select_from_18",
    "build_calibration_for_channel",
]


# ---------------------------------------------------------------------------
# Constants — ported from experiments/tspulse_eval/run_with_direction_calibration.py:60-78
# ---------------------------------------------------------------------------

# Six candidate threshold formulas (limited set controls overfitting on
# channels with only 1-3 events).
CANDIDATE_THRESHOLDS: list[str] = [
    "init512_mean+3σ",
    "global_p90",
    "global_p95",
    "global_p97",
    "global_p99",
    "normal_p99",
]

# Default calibration JSON location — sits next to phm.db under ground/data.
# Path from here (src/ground/phm/algorithm/): up 3 dirnames → src/ground/, then data/.
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "channel_calibration.json",
)


# ---------------------------------------------------------------------------
# Event helper — ported from experiments/metrics/engineering_metrics.py:42-54
# ---------------------------------------------------------------------------

def find_events(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return ground-truth anomaly events as ``[(start, end_exclusive), ...]``.

    ``labels`` is a 1-D 0/1 array.  Consecutive 1s form a single event.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    if labels.sum() == 0:
        return []
    diff = np.diff(np.concatenate([[0], labels, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]  # exclusive end
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# Threshold computation — ported from run_with_direction_calibration.py:204-216
# ---------------------------------------------------------------------------

def compute_threshold(score: np.ndarray, labels: np.ndarray, name: str) -> float:
    """Compute a threshold value by candidate name.

    ``normal_p99`` / ``init512_mean+3σ`` use the first 512 points as a
    normal-segment approximation (system-startup baseline assumption).
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    if name == "init512_mean+3σ":
        init = score[: min(512, len(score))]
        return float(init.mean() + 3 * init.std())
    if name.startswith("global_p"):
        p = float(name.split("_p")[1])
        return float(np.percentile(score, p))
    if name == "normal_p99":
        init = score[: min(512, len(score))]
        return float(np.percentile(init, 99))
    raise ValueError(f"unknown threshold: {name}")


# ---------------------------------------------------------------------------
# Offline evaluation helper
# ---------------------------------------------------------------------------

def _eval_at_threshold(score: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Per-threshold engineering metrics — ported from run_with_direction_calibration.py:219-237.

    ``fa_rate_per_hour`` assumes a 1 Hz sampling rate (matches the NASA-MSL
    eval convention; ``normal_pts / 3600``).  This is for offline threshold
    selection only — it does not run online.
    """
    preds = (score > threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    normal_pts = int(np.sum(labels == 0))
    fa_per_h = fp / (normal_pts / 3600.0) if normal_pts else float("nan")
    events = find_events(labels)
    n_det = sum(1 for s, e in events if preds[s:e].sum() > 0)
    evt_rate = n_det / len(events) if events else float("nan")
    return {
        "recall": recall,
        "precision": precision,
        "fa_rate_per_hour": fa_per_h,
        "event_detection_rate": evt_rate,
    }


# ---------------------------------------------------------------------------
# 18-candidate LOO selection — ported from run_per_channel_select.py:97-145
# ---------------------------------------------------------------------------

def loo_select_from_18(
    scores_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    threshold_names: list[str] | None = None,
    target_evt: float = 0.94,
) -> tuple[str, str, float, dict]:
    """Pick the best (score_type, threshold) pair from 3×6=18 candidates.

    Args:
        scores_dict: ``{"tsp": array, "freq": array, "fusion": array}`` —
            each array already direction-calibrated and MinMax-normalised.
            ``fusion`` is conventionally ``np.maximum(tsp, freq)``.
        labels: 1-D 0/1 ground-truth array.
        threshold_names: candidate threshold names (defaults to
            :data:`CANDIDATE_THRESHOLDS`).
        target_evt: minimum event-detection rate to qualify (default 0.94).

    Returns:
        ``(best_score_key, best_thr_name, best_threshold, detail)``.

    Note:
        The experiment's LOO loop iterates ``leave_idx`` over events but the
        loop body recomputes **full-test** FA each iteration (``leave_idx``
        is unused), so ``loo_fa_mean`` equals the full-test FA.  This
        behaviour is preserved verbatim for numerical consistency with the
        published experiment results.  A true leave-one-out would need
        re-evaluation across all 27 channels.
    """
    if threshold_names is None:
        threshold_names = CANDIDATE_THRESHOLDS
    labels = np.asarray(labels, dtype=int).ravel()

    events = find_events(labels)
    if len(events) == 0:
        thr = compute_threshold(scores_dict["tsp"], labels, "init512_mean+3σ")
        return "tsp", "init512_mean+3σ", thr, {"reason": "no_events"}

    # Step 1: full-test evt + FA for every (score, threshold) candidate.
    cand_full: dict[tuple[str, str], dict] = {}
    for sk, score in scores_dict.items():
        for tn in threshold_names:
            thr = compute_threshold(score, labels, tn)
            e = _eval_at_threshold(score, labels, thr)
            cand_full[(sk, tn)] = {
                "threshold": thr,
                "evt": e["event_detection_rate"],
                "fa": e["fa_rate_per_hour"],
                "recall": e["recall"],
            }

    # Step 2: keep candidates meeting the evt target; if none, keep the max-evt ones.
    qualified = [(k, d) for k, d in cand_full.items() if d["evt"] >= target_evt]
    if not qualified:
        max_evt = max(d["evt"] for d in cand_full.values())
        qualified = [(k, d) for k, d in cand_full.items() if d["evt"] >= max_evt - 1e-9]

    # Step 3: tiebreak by (pseudo-)LOO FA mean.  NOTE: see docstring — this
    # is effectively full-test FA, matching the experiment's behaviour.
    loo_fa: dict[tuple[str, str], list[float]] = {k: [] for k, _ in qualified}
    for _leave_idx in range(len(events)):  # noqa: B007 — preserved per experiment
        for k, _ in qualified:
            sk, _tn = k
            preds = (scores_dict[sk] > cand_full[k]["threshold"]).astype(int)
            normal_pts = int(np.sum(labels == 0))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            loo_fa[k].append(fp / (normal_pts / 3600.0) if normal_pts else 0.0)

    best_key = min(qualified, key=lambda kd: float(np.mean(loo_fa[kd[0]])))[0]
    best_thr = cand_full[best_key]["threshold"]
    return best_key[0], best_key[1], best_thr, {
        "loo_detail": {
            f"{sk}_{tn}": {
                "evt_full": d["evt"],
                "fa_full": d["fa"],
                "loo_fa_mean": float(np.mean(loo_fa[(sk, tn)]))
                if (sk, tn) in loo_fa
                else None,
            }
            for (sk, tn), d in cand_full.items()
        },
        "qualified": [f"{k[0]}_{k[1]}" for k, _ in qualified],
    }


# ---------------------------------------------------------------------------
# Dataclass + JSON container
# ---------------------------------------------------------------------------

@dataclass
class ChannelCalibration:
    """Per-channel calibration record (one entry in the JSON file).

    Attributes:
        flip: whether to invert the score direction (from
            :meth:`DirectionCalibrator.fit`).
        score_type: which score to use — ``"tsp"``, ``"freq"`` or
            ``"fusion"`` (= max(tsp, freq)).
        threshold: numeric anomaly threshold for this channel.
        threshold_name: which candidate formula produced ``threshold``.
        freq_band_mean: STFT band-power mean baseline (only present when
            ``score_type`` involves freq).
        freq_band_std: STFT band-power std baseline (same).
    """

    flip: bool = False
    score_type: str = "tsp"
    threshold: float = 0.5
    threshold_name: str = "global_p99"
    freq_band_mean: list[float] | None = None
    freq_band_std: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChannelCalibration":
        return cls(
            flip=bool(d.get("flip", False)),
            score_type=str(d.get("score_type", "tsp")),
            threshold=float(d.get("threshold", 0.5)),
            threshold_name=str(d.get("threshold_name", "global_p99")),
            freq_band_mean=d.get("freq_band_mean"),
            freq_band_std=d.get("freq_band_std"),
        )


class CalibrationConfig:
    """Loader for ``channel_calibration.json``.

    The file is optional — if absent, every channel returns ``None`` and
    the cascade runs in its default (TSPulse-only, unflipped) mode.  This
    keeps the system fully backward-compatible before the calibration has
    been run.

    Example JSON structure::

        {
          "T-1": {"flip": false, "score_type": "tsp", ...},
          "M-5": {"flip": true,  "score_type": "freq", "freq_band_mean": [...], ...}
        }
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._cal: dict[str, ChannelCalibration] = {}
        self.load()

    def load(self) -> None:
        """(Re)load the JSON.  Missing file → empty config, no error."""
        self._cal = {}
        if not os.path.exists(self.config_path):
            logger.debug(
                "channel calibration %s not found — running uncalibrated",
                self.config_path,
            )
            return
        try:
            with open(self.config_path, encoding="utf-8") as f:
                raw = json.load(f)
            for ch, d in raw.items():
                self._cal[ch] = ChannelCalibration.from_dict(d)
            logger.info(
                "loaded channel calibration for %d channels from %s",
                len(self._cal),
                self.config_path,
            )
        except Exception:
            logger.warning(
                "failed to load channel calibration %s — running uncalibrated",
                self.config_path,
                exc_info=True,
            )
            self._cal = {}

    def reload(self) -> None:
        """Alias for :meth:`load` (hot-reload use case)."""
        self.load()

    def get(self, channel: str) -> ChannelCalibration | None:
        """Return the calibration for ``channel`` or ``None`` if absent."""
        return self._cal.get(channel)

    @property
    def channels(self) -> list[str]:
        return list(self._cal.keys())


# ---------------------------------------------------------------------------
# End-to-end offline calibration for one channel
# ---------------------------------------------------------------------------

def build_calibration_for_channel(
    tsp_score: np.ndarray,
    freq_score: np.ndarray,
    labels: np.ndarray,
    freq_band_mean: np.ndarray | None = None,
    freq_band_std: np.ndarray | None = None,
    target_evt: float = 0.94,
) -> ChannelCalibration:
    """Run the full offline calibration pipeline for one channel.

    Inputs must already be MinMax-normalised to ``[0,1]`` and **not yet
    direction-flipped** — this function applies the flip itself.

    Args:
        tsp_score: MinMax-normalised TSPulse score (1-D).
        freq_score: MinMax-normalised STFT frequency score (1-D), same length.
        labels: 1-D 0/1 ground-truth array.
        freq_band_mean / freq_band_std: the STFT baseline arrays (to embed
            in the output config so the online path can rebuild the
            freq scorer without re-fitting).  Required when the LOO picks
            a freq/fusion score type; ignored otherwise.
        target_evt: LOO event-detection target (default 0.94).

    Returns:
        A :class:`ChannelCalibration` ready to serialise.
    """
    tsp_score = np.asarray(tsp_score, dtype=np.float32).ravel()
    freq_score = np.asarray(freq_score, dtype=np.float32).ravel()
    labels = np.asarray(labels, dtype=int).ravel()

    # Step 1: few-shot direction judge (uses the tsp score; applied to both).
    flip, _detail = DirectionCalibrator.fit(tsp_score, labels)
    tsp_cal = DirectionCalibrator.flip(tsp_score, flip)
    freq_cal = DirectionCalibrator.flip(freq_score, flip)
    fusion_cal = np.maximum(tsp_cal, freq_cal).astype(np.float32)

    scores_dict = {"tsp": tsp_cal, "freq": freq_cal, "fusion": fusion_cal}

    # Step 2: 18-candidate LOO selection.
    best_sk, best_tn, best_thr, _detail = loo_select_from_18(
        scores_dict, labels, target_evt=target_evt
    )

    # Step 3: only carry the freq baseline when it's actually used.
    need_freq = best_sk in ("freq", "fusion")
    return ChannelCalibration(
        flip=flip,
        score_type=best_sk,
        threshold=float(best_thr),
        threshold_name=best_tn,
        freq_band_mean=(
            [float(x) for x in np.asarray(freq_band_mean).ravel()] if need_freq else None
        ),
        freq_band_std=(
            [float(x) for x in np.asarray(freq_band_std).ravel()] if need_freq else None
        ),
    )
