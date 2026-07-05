"""In-memory early-warning (预警) store with lifecycle state machine.

Each warning entry represents a *predicted* anomaly — the ground-side
forecast+detect pipeline predicted that a future window would exceed the
threshold.  When later measured data arrives we verify whether the
prediction was accurate, updating the entry's ``status``:

    pending  → (predicted to exceed, not yet verifiable)
    confirmed → (later measured data in that window DID exceed)
    false    → (later measured data in that window did NOT exceed)

Two parallel report streams are therefore surfaced to the UI:

    type="measured" : confirmed anomalies (from alert_store)
    type="predicted": forecast-derived warnings (this store)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..config import ANOMALY_THRESHOLD


@dataclass
class WarningEntry:
    channel: str
    # Time range (epoch seconds) that the prediction covers — used to
    # match against later measured data.
    predict_start: float
    predict_end: float
    max_predict_score: float
    created_at: float = field(default_factory=time.time)
    # lifecycle: pending | confirmed | false
    status: str = "pending"
    # When verified, the measured max score inside the prediction window.
    verified_max_score: float | None = None
    verified_at: float | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "predict_start": self.predict_start,
            "predict_end": self.predict_end,
            "max_predict_score": self.max_predict_score,
            "created_at": self.created_at,
            "status": self.status,
            "verified_max_score": self.verified_max_score,
            "verified_at": self.verified_at,
            "message": self.message,
            "type": "predicted",
        }


class WarningStore:
    """Thread-safe warning registry with verification support."""

    def __init__(self, max_size: int = 200) -> None:
        self._entries: list[WarningEntry] = []
        self._lock = threading.Lock()
        self._max_size = max_size

    # -- create -------------------------------------------------------------

    def add_pending(
        self,
        channel: str,
        predict_start: float,
        predict_end: float,
        max_predict_score: float,
        message: str = "",
    ) -> WarningEntry | None:
        """Create a new pending warning if one for the same (channel,
        window) is not already present.  Returns the entry, or None if a
        duplicate was skipped."""
        with self._lock:
            # De-dupe: skip if an active pending entry already covers an
            # overlapping window for this channel.
            for e in self._entries:
                if (
                    e.channel == channel
                    and e.status == "pending"
                    and self._overlaps(e, predict_start, predict_end)
                ):
                    return None
            entry = WarningEntry(
                channel=channel,
                predict_start=predict_start,
                predict_end=predict_end,
                max_predict_score=float(max_predict_score),
                message=message or f"预测异常分数 {max_predict_score:.3f} > {ANOMALY_THRESHOLD}",
            )
            self._entries.append(entry)
            if len(self._entries) > self._max_size:
                self._entries = self._entries[-self._max_size:]
            return entry

    # -- verify -------------------------------------------------------------

    def verify(self, channel: str, measured_scores_by_time: list[tuple[float, float]]) -> int:
        """Walk pending entries for ``channel`` and verify them against
        newly arrived measured ``(timestamp, score)`` samples.

        Returns the number of entries whose status changed.
        """
        if not measured_scores_by_time:
            return 0
        changed = 0
        now = time.time()
        with self._lock:
            for e in self._entries:
                if e.channel != channel or e.status != "pending":
                    continue
                # Only verify once the prediction window has elapsed enough
                # that measured data should have arrived (>= predict_end).
                if now < e.predict_end:
                    continue
                in_window = [
                    s for (t, s) in measured_scores_by_time
                    if e.predict_start <= t <= e.predict_end
                ]
                if not in_window:
                    continue
                mx = float(max(in_window))
                e.verified_max_score = mx
                e.verified_at = now
                if mx > ANOMALY_THRESHOLD:
                    e.status = "confirmed"
                    e.message = (
                        f"预测准确：实测异常分数 {mx:.3f} > {ANOMALY_THRESHOLD}"
                    )
                else:
                    e.status = "false"
                    e.message = (
                        f"预测误报：实测异常分数 {mx:.3f} ≤ {ANOMALY_THRESHOLD}"
                    )
                changed += 1
        return changed

    # -- read ---------------------------------------------------------------

    def all(self) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries]

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries[-limit:]]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _overlaps(e: WarningEntry, s: float, en: float) -> bool:
        return not (en < e.predict_start or s > e.predict_end)


__all__ = ["WarningStore", "WarningEntry"]
