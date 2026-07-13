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

# Candidate threshold formulas used in LOO selection.  These six are all
# computed on the test segment's score distribution.
#
# NOTE: ``train_target_fa5`` / ``train_target_fa20`` (leak-free formulas
# derived from the training segment) are implemented in
# :func:`compute_threshold` and were evaluated as calibration candidates
# but are **not** included in the default set — on NASA-MSL/SMAP the
# training-segment score distribution systematically differs from the
# test normal-segment distribution (TSPulse reconstruction error is not
# stationary across train/test), so train-derived thresholds inflate the
# calibration-vs-online gap and raise the online FA rate.  See
# ``docs/项目现状.md`` Day15 P1 record for the experiment that established
# this.  The implementation is kept so the candidate can be re-evaluated
# on future datasets where the stationarity assumption holds.
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

def compute_threshold(
    score: np.ndarray,
    labels: np.ndarray,
    name: str,
    train_score: np.ndarray | None = None,
    sr_hz: float = 1.0,
) -> float:
    """Compute a threshold value by candidate name.

    Legacy formulas (``init512_mean+3σ`` / ``normal_p99`` / ``global_p*``)
    are computed on the test ``score`` array and are kept for backward
    comparison.  The ``train_target_fa<N>`` formulas invert a target FA/h
    rate from the *training* segment's score distribution: given a target
    FA/h ``N``, the threshold is the ``(1 − N/(sr·3600))`` quantile of
    ``train_score``.  This is leak-free (the training segment is known to
    be all-normal) and the FA/h target carries the same business meaning
    across datasets.
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
    if name.startswith("train_target_fa"):
        target_fa = float(name.split("_fa")[1])
        if train_score is None or len(train_score) == 0:
            # No training scores available — fall back to a conservative
            # test-side quantile so the candidate still participates in LOO.
            return float(np.percentile(score, 99))
        p = (1.0 - target_fa / (sr_hz * 3600.0)) * 100.0
        return float(np.percentile(train_score, p))
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
    train_scores_dict: dict[str, np.ndarray] | None = None,
    sr_hz: float = 1.0,
) -> tuple[str, str, float, dict]:
    """Pick the best (score_type, threshold) pair from the candidate set.

    Args:
        scores_dict: ``{"tsp": array, "freq": array, "fusion": array}`` —
            each array already direction-calibrated and normalised.
            ``fusion`` is conventionally ``np.maximum(tsp, freq)``.
        labels: 1-D 0/1 ground-truth array (test segment).
        threshold_names: candidate threshold names (defaults to
            :data:`CANDIDATE_THRESHOLDS`).
        target_evt: minimum event-detection rate to qualify (default 0.94).
        train_scores_dict: optional ``{"tsp": array, "freq": array,
            "fusion": array}`` of training-segment scores (same
            direction-calibration as ``scores_dict``).  Required for the
            ``train_target_fa*`` formulas; when None those formulas fall
            back to a test-side quantile inside :func:`compute_threshold`.
        sr_hz: sampling rate in Hz — used by the ``train_target_fa*``
            formulas to convert FA/h to a quantile.

    Returns:
        ``(best_score_key, best_thr_name, best_threshold, detail)``.
    """
    if threshold_names is None:
        threshold_names = CANDIDATE_THRESHOLDS
    labels = np.asarray(labels, dtype=int).ravel()

    events = find_events(labels)
    if len(events) == 0:
        thr = compute_threshold(
            scores_dict["tsp"], labels, "init512_mean+3σ",
            train_score=train_scores_dict.get("tsp") if train_scores_dict else None,
            sr_hz=sr_hz,
        )
        return "tsp", "init512_mean+3σ", thr, {"reason": "no_events"}

    # Step 1: full-test evt + FA for every (score, threshold) candidate.
    # For train_target_fa* formulas the threshold is derived from the
    # training-segment score distribution (leak-free); the test evt/FA is
    # still evaluated on the test segment but only used for selection.
    cand_full: dict[tuple[str, str], dict] = {}
    for sk, score in scores_dict.items():
        train_score = (
            train_scores_dict.get(sk) if train_scores_dict is not None else None
        )
        for tn in threshold_names:
            thr = compute_threshold(score, labels, tn, train_score=train_score, sr_hz=sr_hz)
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

    # Step 3: tiebreak by (pseudo-)LOO FA mean.  The experiment's LOO loop
    # iterates ``leave_idx`` over events but recomputes full-test FA each
    # iteration (``leave_idx`` unused), so this equals full-test FA.  Kept
    # verbatim for numerical consistency with the published experiment.
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
        freq_z_min / freq_z_max: reference range of the band z-score over
            the training segment, used by
            :class:`FreqFeatureExtractor.transform` to map scores onto
            ``[0, 1]``.  Required when ``score_type`` involves freq; absent
            on older JSONs forces the legacy per-call MinMax fallback.
    """

    flip: bool = False
    score_type: str = "tsp"
    threshold: float = 0.5
    threshold_name: str = "global_p99"
    freq_band_mean: list[float] | None = None
    freq_band_std: list[float] | None = None
    freq_z_min: float | None = None
    freq_z_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChannelCalibration":
        z_min = d.get("freq_z_min")
        z_max = d.get("freq_z_max")
        return cls(
            flip=bool(d.get("flip", False)),
            score_type=str(d.get("score_type", "tsp")),
            threshold=float(d.get("threshold", 0.5)),
            threshold_name=str(d.get("threshold_name", "global_p99")),
            freq_band_mean=d.get("freq_band_mean"),
            freq_band_std=d.get("freq_band_std"),
            freq_z_min=float(z_min) if z_min is not None else None,
            freq_z_max=float(z_max) if z_max is not None else None,
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
    freq_z_min: float | None = None,
    freq_z_max: float | None = None,
    target_evt: float = 0.94,
    tsp_train_score: np.ndarray | None = None,
    freq_train_score: np.ndarray | None = None,
    sr_hz: float = 1.0,
) -> ChannelCalibration:
    """Run the full offline calibration pipeline for one channel.

    Inputs must already be normalised to ``[0,1]`` (TSPulse clip-normalised;
    freq score mapped via the training-segment z-score reference) and **not
    yet direction-flipped** — this function applies the flip itself.

    Args:
        tsp_score: clip-normalised TSPulse score (1-D), in ``[0,1]``.
        freq_score: STFT frequency score (1-D), already mapped onto
            ``[0,1]`` via the training-segment z_min/z_max reference so it
            matches the online :meth:`FreqFeatureExtractor.transform` scale.
        labels: 1-D 0/1 ground-truth array.
        freq_band_mean / freq_band_std: the STFT baseline arrays (to embed
            in the output config so the online path can rebuild the
            freq scorer without re-fitting).  Required when the LOO picks
            a freq/fusion score type; ignored otherwise.
        freq_z_min / freq_z_max: the z-score reference range (to embed
            alongside the band baseline).  Required for freq/fusion types.
        target_evt: LOO event-detection target (default 0.94).
        tsp_train_score / freq_train_score: optional training-segment
            scores (raw, not yet direction-flipped — the flip is applied
            here consistently).  When provided, the ``train_target_fa*``
            threshold candidates become available and are preferred over
            the legacy test-side formulas (leak-free threshold selection).
        sr_hz: sampling rate in Hz (used by train_target_fa* formulas).

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

    # Build the training-side score dict (same flip + fusion as test) so the
    # train_target_fa* candidates can be computed leak-free.
    train_scores_dict: dict[str, np.ndarray] | None = None
    if tsp_train_score is not None:
        tsp_tr_cal = DirectionCalibrator.flip(
            np.asarray(tsp_train_score, dtype=np.float32).ravel(), flip
        )
        freq_tr_cal = (
            DirectionCalibrator.flip(
                np.asarray(freq_train_score, dtype=np.float32).ravel(), flip
            )
            if freq_train_score is not None
            else tsp_tr_cal  # fall back to tsp if freq not provided
        )
        fusion_tr_cal = np.maximum(tsp_tr_cal, freq_tr_cal).astype(np.float32)
        train_scores_dict = {"tsp": tsp_tr_cal, "freq": freq_tr_cal, "fusion": fusion_tr_cal}

    # Step 2: candidate selection (8 formulas now, including train_target_fa*).
    best_sk, best_tn, best_thr, _detail = loo_select_from_18(
        scores_dict, labels, target_evt=target_evt,
        train_scores_dict=train_scores_dict, sr_hz=sr_hz,
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
        freq_z_min=float(freq_z_min) if need_freq and freq_z_min is not None else None,
        freq_z_max=float(freq_z_max) if need_freq and freq_z_max is not None else None,
    )
