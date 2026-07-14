"""In-memory early-warning (预警) store with lifecycle state machine.

Each warning entry represents a *predicted* anomaly — the ground-side
forecast+detect pipeline predicted that a future window would exceed the
threshold.  When later measured data arrives we verify whether the
prediction was accurate, updating the entry's ``verify_status``:

    pending  → (predicted to exceed, not yet verifiable)
    confirmed → (later measured data in that window DID exceed)
    false    → (later measured data in that window did NOT exceed)
    unverifiable → (window elapsed but no measured data arrived)

Four-dimension verdict system (see spec §3.2):
    - model_alert    : detection model flag (implicit, from score > threshold)
    - verify_status  : automatic prediction verification (this store)
    - llm_verdict    : LLM diagnosis verdict (real/false_alarm/uncertain)
    - human_verdict  : manual human annotation (real/false_alarm/uncertain)

    final_status = human_verdict ?? llm_verdict ?? verify_status

Two parallel report streams are therefore surfaced to the UI:

    type="measured" : confirmed anomalies (from alert_store)
    type="predicted": forecast-derived warnings (this store)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..config import ANOMALY_THRESHOLD

_VALID_VERDICTS = frozenset({"real", "false_alarm", "uncertain"})


def compute_final_status(
    verify_status: str,
    llm_verdict: str | None,
    human_verdict: str | None,
) -> str:
    """Derive the display status from the four-dimension verdicts.

    Priority: human_verdict > llm_verdict > verify_status.
    """
    if human_verdict is not None:
        return human_verdict
    if llm_verdict is not None:
        return llm_verdict
    return verify_status


@dataclass
class WarningEntry:
    channel: str
    # Time range (epoch seconds) that the prediction covers — used to
    # match against later measured data.
    predict_start: float
    predict_end: float
    max_predict_score: float
    created_at: float = field(default_factory=time.time)
    # Unique id assigned by WarningStore.add_pending (0 = not yet registered).
    id: int = 0
    # Dimension 1: detection model flag (implicit from score > threshold).
    # Dimension 2: prediction verification status.
    #     pending | confirmed | false | unverifiable
    verify_status: str = "pending"
    # When verified, the measured max score inside the prediction window.
    verified_max_score: float | None = None
    verified_at: float | None = None
    # Dimension 3: LLM diagnosis verdict (real/false_alarm/uncertain, None = not diagnosed).
    llm_verdict: str | None = None
    # Dimension 4: human annotation verdict (real/false_alarm/uncertain, None = not annotated).
    human_verdict: str | None = None
    message: str = ""

    # -- backward-compat: `status` property mirrors verify_status --------

    @property
    def status(self) -> str:
        return self.verify_status

    @status.setter
    def status(self, value: str) -> None:
        self.verify_status = value

    def to_dict(self) -> dict:
        final = compute_final_status(self.verify_status, self.llm_verdict, self.human_verdict)
        return {
            "channel": self.channel,
            "predict_start": self.predict_start,
            "predict_end": self.predict_end,
            "max_predict_score": self.max_predict_score,
            "created_at": self.created_at,
            "id": self.id,
            "verify_status": self.verify_status,
            "status": self.verify_status,  # backward compat
            "verified_max_score": self.verified_max_score,
            "verified_at": self.verified_at,
            "llm_verdict": self.llm_verdict,
            "human_verdict": self.human_verdict,
            "final_status": final,
            "message": self.message,
            "type": "predicted",
        }


class WarningStore:
    """Thread-safe warning registry with verification support."""

    def __init__(self, max_size: int = 200) -> None:
        self._entries: list[WarningEntry] = []
        self._lock = threading.Lock()
        self._max_size = max_size
        self._next_id = 1

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
                    and e.verify_status == "pending"
                    and self._overlaps(e, predict_start, predict_end)
                ):
                    return None
            entry = WarningEntry(
                channel=channel,
                predict_start=predict_start,
                predict_end=predict_end,
                max_predict_score=float(max_predict_score),
                message=message or f"预测异常分数 {max_predict_score:.3f} > {ANOMALY_THRESHOLD}",
                id=self._next_id,
            )
            self._next_id += 1
            self._entries.append(entry)
            if len(self._entries) > self._max_size:
                self._entries = self._entries[-self._max_size:]
            return entry

    # -- verify -------------------------------------------------------------

    def verify(self, channel: str, measured_scores_by_time: list[tuple[float, float]]) -> int:
        """Walk pending entries for ``channel`` and verify them against
        newly arrived measured ``(timestamp, score)`` samples.

        If the prediction window has elapsed but no measured data is
        available, the entry is marked ``unverifiable``.

        Returns the number of entries whose status changed.
        """
        changed = 0
        now = time.time()
        with self._lock:
            for e in self._entries:
                if e.channel != channel or e.verify_status != "pending":
                    continue
                # Only verify once the prediction window has elapsed enough
                # that measured data should have arrived (>= predict_end).
                if now < e.predict_end:
                    continue
                in_window = [
                    s for (t, s) in measured_scores_by_time
                    if e.predict_start <= t <= e.predict_end
                ]
                e.verified_at = now
                if not in_window:
                    e.verify_status = "unverifiable"
                    e.message = "预测窗口已过但无实测数据核验"
                    changed += 1
                    continue
                mx = float(max(in_window))
                e.verified_max_score = mx
                if mx > ANOMALY_THRESHOLD:
                    e.verify_status = "confirmed"
                    e.message = (
                        f"预测准确：实测异常分数 {mx:.3f} > {ANOMALY_THRESHOLD}"
                    )
                else:
                    e.verify_status = "false"
                    e.message = (
                        f"预测误报：实测异常分数 {mx:.3f} ≤ {ANOMALY_THRESHOLD}"
                    )
                changed += 1
        return changed

    # -- verdict ---------------------------------------------------------------

    def set_verdict(self, entry_id: int, verdict_type: str, value: str) -> bool:
        """Set a verdict (llm or human) on an entry by id.

        Args:
            entry_id: the WarningEntry.id to update.
            verdict_type: ``"llm"`` or ``"human"``.
            value: ``"real"``, ``"false_alarm"``, or ``"uncertain"``.

        Returns True if the entry was found and updated.
        """
        if value not in _VALID_VERDICTS:
            return False
        with self._lock:
            for e in self._entries:
                if e.id == entry_id:
                    if verdict_type == "llm":
                        e.llm_verdict = value
                    elif verdict_type == "human":
                        e.human_verdict = value
                    else:
                        return False
                    return True
            return False

    def get_by_id(self, entry_id: int) -> WarningEntry | None:
        with self._lock:
            for e in self._entries:
                if e.id == entry_id:
                    return e
            return None

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


__all__ = ["WarningStore", "WarningEntry", "compute_final_status"]
