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

from ..algorithm import BaseDetector
from ..config import ANOMALY_THRESHOLD, FORECAST_PREDICTION_LENGTH
from ..database import RingBuffer
from ..database.warning_store import WarningStore
from .forecast_service import ForecastService

logger = logging.getLogger(__name__)


class WarningService:
    """Combined forecast+detect early-warning state machine."""

    def __init__(
        self,
        ring: RingBuffer,
        warnings: WarningStore,
        forecast_service: ForecastService,
        detector: BaseDetector | None = None,
        threshold: float = ANOMALY_THRESHOLD,
    ) -> None:
        self.ring = ring
        self.warnings = warnings
        self.forecast_service = forecast_service
        self._detector: BaseDetector | None = detector
        self._detector_init_failed = False
        self.threshold = threshold
        # Cache latest predict scores per channel for chart display
        self._latest_predict_scores: dict[str, dict] = {}

    # -- detector lazy load -------------------------------------------------

    def _get_detector(self) -> BaseDetector | None:
        if self._detector is not None:
            return self._detector
        if self._detector_init_failed:
            return None
        try:
            from ..algorithm import AnomalyDetector
            self._detector = AnomalyDetector(device="cpu")
        except Exception as e:
            logger.warning("Failed to load ground TSPulse detector: %s", e)
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
        # 1. Pull measured raw values from the ring buffer
        entries = self.ring.raw_block_entries(channel, block_size)
        if len(entries) < 10:
            return None
        raw_values = np.array([e["raw"] for e in entries], dtype=np.float32)
        last_ts = entries[-1]["received_at"]

        # 2. Forecast
        fc_result = self.forecast_service.forecast(raw_values.tolist())
        if "prediction" not in fc_result:
            return None
        prediction = np.array(fc_result["prediction"], dtype=np.float32)

        # 3. Combined detect
        detector = self._get_detector()
        if detector is None:
            return None
        try:
            combined = np.concatenate([raw_values, prediction]).astype(np.float32)
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
        if len(entries) >= 2:
            interval = (entries[-1]["received_at"] - entries[0]["received_at"]) / max(1, len(entries) - 1)
        else:
            interval = 0.02
        predict_timestamps = [last_ts + (i + 1) * interval for i in range(len(predict_scores))]
        self._latest_predict_scores[channel] = {
            "timestamps": predict_timestamps,
            "scores": predict_scores.tolist(),
            "predict_start": last_ts,
            "predict_end": predict_timestamps[-1] if predict_timestamps else last_ts,
        }

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

    def clear(self) -> None:
        self.warnings.clear()
        self._latest_predict_scores.clear()


__all__ = ["WarningService"]
