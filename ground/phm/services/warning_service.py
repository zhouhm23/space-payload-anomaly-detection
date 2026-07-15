"""Ground-side combined forecast+detect early-warning service.

Implements the user-specified ground data-processing flow:

    实测块 telemetry[N]
      ① TTM-R3 预测 → prediction[96]
      ② 拼接联合序列 combined = telemetry[N] + prediction[96]
      ③ TSPulse 联合检测 scores_all = detect(combined)   # 长 N+96
      ④ 只取预测段 predict_scores = scores_all[N:]
      ⑤ max(predict_scores) > threshold → 新增预警 (status=pending)
      ⑥ 后续实测覆盖预测区间 → 取该区间天基实测分数核验
         实测超阈值 → status=confirmed (真实/预测准确)
         否则       → status=false     (虚报/预测误报)

The telemetry chart's anomaly-score curve is **not** touched by this
service — it keeps using the space-side scores stored in the ring buffer,
so forecast data never contaminates the measured anomaly display.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..algorithm import BaseDetector, CascadeDetector
from ..algorithm.calibration_config import CalibrationConfig
from ..algorithm.classic_filter import ClassicFilter
from ..algorithm.joint_detector import co_anomaly_consensus
from ..algorithm.physical_constraint import ConstraintConfig, PhysicalConstraint
from ..config import (
    ANOMALY_THRESHOLD, FORECAST_PREDICTION_LENGTH,
    L1_CONSTANT_STD, L1_SIGMA_K, L1_IQR_FACTOR,
    L3_CONSTANT_STD, L3_RANGE_BOOST, L3_RATE_BOOST,
)
from ..database import RingBuffer, SQLiteStore
from ..database.warning_store import WarningStore
from .forecast_service import ForecastService
from .tree_utils import get_sensor_to_folder, _find_node_by_id

logger = logging.getLogger(__name__)

# Joint (co-anomaly) alert threshold: triggers when the fraction of sibling
# channels exceeding their per-channel threshold is above this value.
# 0.5 = majority consensus.  Stored here (not in config.py) because joint
# detection is an additive layer that can be ignored if no folder has ≥2
# sensors.
_JOINT_ALERT_THRESHOLD = 0.5


class WarningService:
    """Combined forecast+detect early-warning state machine."""

    def __init__(
        self,
        ring: RingBuffer,
        warnings: WarningStore,
        forecast_service: ForecastService,
        sqlite: SQLiteStore | None = None,
        detector: BaseDetector | None = None,
        threshold: float = ANOMALY_THRESHOLD,
    ) -> None:
        self.ring = ring
        self.warnings = warnings
        self.forecast_service = forecast_service
        self.sqlite = sqlite
        self._detector: BaseDetector | None = detector
        self._detector_init_failed = False
        self.threshold = threshold
        # Cache latest predict scores per channel for chart display
        self._latest_predict_scores: dict[str, dict] = {}
        # Cache latest cascade output per channel (for /api/detection)
        self._latest_cascade: dict[str, object] = {}
        # Remember the last raw timestamp evaluated per channel so we can
        # skip re-evaluation when no new raw has arrived (the eval thread
        # fires every 1s but new data only arrives every ~2s via auto-poll).
        self._last_eval_ts: dict[str, float] = {}

    # -- detector lazy load -------------------------------------------------

    def _get_detector(self) -> BaseDetector | None:
        if self._detector is not None:
            return self._detector
        if self._detector_init_failed:
            return None
        try:
            from ..algorithm import AnomalyDetector
            base_detector = AnomalyDetector(device="cpu")
            # Wrap in three-layer cascade
            classic = ClassicFilter(
                constant_std=L1_CONSTANT_STD,
                sigma_k=L1_SIGMA_K,
                iqr_factor=L1_IQR_FACTOR,
            )
            constraint = PhysicalConstraint(
                ConstraintConfig(
                    constant_std=L3_CONSTANT_STD,
                    range_boost=L3_RANGE_BOOST,
                    rate_boost=L3_RATE_BOOST,
                )
            )
            # Load per-channel offline calibration (direction flip + score-type
            # selection + threshold).  Missing JSON ⇒ empty config, cascade
            # falls back to default TSPulse-only path (backward compatible).
            calibration = CalibrationConfig()
            self._detector = CascadeDetector(
                detector=base_detector,
                classic=classic,
                constraint=constraint,
                calibration_config=calibration,
            )
            logger.info(
                "Ground cascade detector ready (TSPulse + L1 + L3, %d calibrated channels)",
                len(calibration.channels),
            )
        except Exception as e:
            logger.warning("Failed to load ground cascade detector: %s", e)
            self._detector_init_failed = True
            return None
        return self._detector

    # -- main entry: called after each telemetry poll ----------------------

    def evaluate_channel(self, channel: str, block_size: int = 512) -> dict | None:
        """Run the combined pipeline for one channel and (maybe) create a
        new pending warning.  Returns the warning dict, or None.

        Side effect: also triggers verification of older pending warnings
        for this channel against any newly arrived measured data.
        """
        # 1. Pull measured raw values from the ring buffer.
        # Fetch 2× block: the earlier half serves as overlap context for the
        # pipeline (without it, short blocks of slowly-varying channels
        # produce near-zero scores — see AnomalyDetector.detect docstring).
        entries = self.ring.raw_block_entries(channel, block_size * 2)
        if len(entries) < 10:
            return None
        # Split: earlier half = context, later half = target raw_values
        split = max(block_size, len(entries) - block_size)
        context_arr = np.array([e["raw"] for e in entries[:split]], dtype=np.float32) if len(entries) > block_size else None
        target_entries = entries[split:] if len(entries) > block_size else entries
        raw_values = np.array([e["raw"] for e in target_entries], dtype=np.float32)
        last_ts = entries[-1]["received_at"]

        # Skip if no new raw has arrived since the last evaluation.  The eval
        # thread fires every 1s but new data arrives only every ~2s (auto-
        # poll), so without this guard each pred interval is re-evaluated
        # 2-3x — wasting TTM-R3 + TSPulse inference and making the predicted
        # anomaly score visibly jump until raw locks it in.
        if self._last_eval_ts.get(channel) == last_ts:
            return None
        self._last_eval_ts[channel] = last_ts

        # 2. Forecast
        fc_result = self.forecast_service.forecast(raw_values.tolist())
        if "prediction" not in fc_result:
            return None
        prediction = np.array(fc_result["prediction"], dtype=np.float32)

        # 3. Combined detect (three-layer cascade)
        detector = self._get_detector()
        if detector is None:
            return None
        try:
            combined = np.concatenate([raw_values, prediction]).astype(np.float32)
            if isinstance(detector, CascadeDetector):
                cascade_out = detector.detect_with_layers(
                    combined, channel=channel, context=context_arr
                )
                scores_all = cascade_out.final_scores
                self._latest_cascade[channel] = cascade_out
                if self.sqlite is not None:
                    self.sqlite.enqueue_detection(channel, last_ts, cascade_out)
            else:
                scores_all = detector.detect(combined)
        except Exception:
            logger.warning("Ground combined detect failed for %s", channel, exc_info=True)
            return None

        # 4. Predict-segment scores only
        n = len(raw_values)
        predict_scores = scores_all[n:]
        if len(predict_scores) == 0:
            return None
        max_pred = float(np.nanmax(predict_scores)) if len(predict_scores) else 0.0

        # 4b. Cache predict scores with timestamps for chart display
        #     Predict timestamps continue the raw grid: last_ts + (i+1)*interval.
        #     Using the SAME arithmetic as raw (interval from actual entries)
        #     ensures pred lands on the same quantum grid as raw after
        #     quantisation, so UPSERT merges them into one row.
        if len(entries) >= 2:
            interval = (entries[-1]["received_at"] - entries[0]["received_at"]) / max(1, len(entries) - 1)
        else:
            interval = 0.02
        pred_timestamps = [last_ts + (i + 1) * interval for i in range(len(prediction))]
        predict_start_ts = pred_timestamps[0] if pred_timestamps else last_ts
        predict_end_ts = pred_timestamps[-1] if pred_timestamps else last_ts
        self._latest_predict_scores[channel] = {
            "timestamps": pred_timestamps[:len(predict_scores)],
            "scores": predict_scores.tolist(),
            "predict_start": predict_start_ts,
            "predict_end": predict_end_ts,
        }

        # Persist predicted values + predicted scores to SQLite
        #     Pass explicit timestamps so SQLiteStore doesn't recompute them
        #     via a different float path (which caused raw/ped row splitting).
        if self.sqlite is not None:
            self.sqlite.enqueue_predictions(
                channel=channel,
                origin_ts=last_ts,
                predict_start=predict_start_ts,
                predict_end=predict_end_ts,
                prediction=prediction.tolist(),
                predict_scores=predict_scores.tolist(),
                model=fc_result.get("model"),
                timestamps=pred_timestamps,
            )

        # 5. Emit pending warning if over threshold
        created = None
        if max_pred > self.threshold:
            # Estimate sample interval from the measured block
            if len(entries) >= 2:
                interval = (entries[-1]["received_at"] - entries[0]["received_at"]) / max(1, len(entries) - 1)
            else:
                interval = 0.02
            predict_start = last_ts
            predict_end = last_ts + len(prediction) * interval
            entry = self.warnings.add_pending(
                channel=channel,
                predict_start=predict_start,
                predict_end=predict_end,
                max_predict_score=max_pred,
                raw_snapshot=raw_values.tolist() if hasattr(raw_values, 'tolist') else list(raw_values),
                pred_snapshot=prediction.tolist() if hasattr(prediction, 'tolist') else list(prediction),
                score_snapshot=predict_scores.tolist() if hasattr(predict_scores, 'tolist') else list(predict_scores),
            )
            if entry is not None:
                created = entry.to_dict()

        # 6. Verify older pending warnings with newly arrived measured data
        measured_by_time = [
            (e["received_at"], e["score"] if e.get("score") is not None else 0.0)
            for e in entries
        ]
        self.warnings.verify(channel, measured_by_time)

        return created

    # -- read --------------------------------------------------------------

    def list(self, limit: int = 50) -> list[dict]:
        return self.warnings.recent(limit)

    def get_latest_predict_scores(self, channel: str) -> dict | None:
        """Return cached predict scores for a channel, or None."""
        return self._latest_predict_scores.get(channel)

    def get_latest_cascade(self, channel: str):
        """Return the latest CascadeOutput for a channel, or None."""
        return self._latest_cascade.get(channel)

    # -- joint (cross-channel) detection -----------------------------------

    def _get_channel_threshold(self, channel: str) -> float:
        """Return the calibrated threshold for a channel (fallback 0.5)."""
        detector = self._get_detector()
        if detector is None:
            return self.threshold
        cal_cfg = getattr(detector, "calibration_config", None)
        if cal_cfg is not None:
            cal = cal_cfg.get(channel)
            if cal is not None and cal.threshold is not None:
                return float(cal.threshold)
        return self.threshold

    def evaluate_all_folders(self, device_tree: list) -> list[dict]:
        """Run co-anomaly consensus for every folder with ≥2 channels.

        Called after the parallel per-channel eval cycle.  Reads the cached
        per-channel predict scores (``_latest_predict_scores``), groups
        channels by their device-tree folder, and computes the co-anomaly
        consensus for each group.  Returns a list of joint-alert dicts for
        folders whose consensus exceeds ``_JOINT_ALERT_THRESHOLD``.

        Each alert dict has the shape consumed by ``_emit_joint_alert``::

            {"folder_id", "folder_name", "joint_score", "channels": [...],
             "score_snapshot": {...}, "alert_ts": float}
        """
        if not device_tree:
            return []

        folder_map = get_sensor_to_folder(device_tree)  # {channel: folder_id}
        if not folder_map:
            return []

        # Group evaluated channels by folder.
        folder_channels: dict[str, list[str]] = {}
        for ch, cached in self._latest_predict_scores.items():
            fid = folder_map.get(ch)
            if fid and cached and cached.get("scores"):
                folder_channels.setdefault(fid, []).append(ch)

        alerts: list[dict] = []
        for fid, channels in folder_channels.items():
            if len(channels) < 2:
                continue  # consensus needs ≥2 siblings

            scores = {
                ch: np.asarray(self._latest_predict_scores[ch]["scores"], dtype=np.float32)
                for ch in channels
            }
            thresholds = {ch: self._get_channel_threshold(ch) for ch in channels}
            joint = co_anomaly_consensus(scores, thresholds)
            if len(joint) == 0:
                continue

            joint_max = float(np.nanmax(joint))
            if joint_max <= _JOINT_ALERT_THRESHOLD:
                continue

            # Resolve folder name for display ("SUB:<name>").
            folder_node = _find_node_by_id(device_tree, fid)
            folder_name = folder_node.get("name", fid) if folder_node else fid

            # Per-channel contribution at the peak point.
            peak_idx = int(np.nanargmax(joint))
            contributions = {}
            for ch in channels:
                s = scores[ch]
                thr = thresholds[ch]
                peak_score = float(s[peak_idx]) if peak_idx < len(s) else 0.0
                contributions[ch] = {
                    "score": peak_score,
                    "threshold": thr,
                    "exceeds": peak_score > thr,
                }

            import time as _time
            alerts.append({
                "folder_id": fid,
                "folder_name": folder_name,
                "joint_score": joint_max,
                "channels": channels,
                "contributions": contributions,
                "joint_curve": joint.tolist(),
                "alert_ts": _time.time(),
            })

        return alerts

    def _emit_joint_alert(self, alert: dict) -> None:
        """Persist a joint alert to SQLite alert_records.

        Uses ``channel="SUB:<folder_name>"`` and ``alert_type="joint"`` so
        the frontend (which renders any channel name) displays it without
        UI changes.  The per-channel contributions are stored in
        ``score_snapshot`` for LLM diagnosis context.
        """
        if self.sqlite is None:
            return
        channel = f"SUB:{alert['folder_name']}"
        self.sqlite.enqueue_alert({
            "channel": channel,
            "alert_type": "joint",
            "score": alert["joint_score"],
            "message": f"子系统联合告警: {alert['folder_name']} "
                       f"({len(alert['channels'])}通道共识 "
                       f"{alert['joint_score']:.0%})",
            "created_at": alert["alert_ts"],
            "status": "active",
            "score_snapshot": {
                "joint_curve": alert["joint_curve"],
                "contributions": alert["contributions"],
                "channels": alert["channels"],
            },
        })

    def clear(self) -> None:
        self.warnings.clear()
        self._latest_predict_scores.clear()
        self._latest_cascade.clear()


__all__ = ["WarningService"]
