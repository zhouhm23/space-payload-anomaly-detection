"""In-memory alert store.

Alerts are **confirmed anomalies** — i.e. measured telemetry whose anomaly
score (from the space-side TSPulse) exceeded the threshold.  They are the
ground-truth "实报" (measured report) stream, distinct from forecast-derived
"预警" (early warnings) which live in ``warning_store.py``.
"""

from __future__ import annotations

import threading
from collections import deque

from ..config import ANOMALY_THRESHOLD


class AlertStore:
    """Thread-safe bounded alert queue."""

    def __init__(self, max_size: int = 500) -> None:
        self._alerts: deque[dict] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, alert: dict) -> None:
        """Append an alert dict (shape mirrors legacy AlertPacket)."""
        with self._lock:
            self._alerts.append(alert)

    def extend(self, alerts: list[dict]) -> None:
        with self._lock:
            for a in alerts:
                self._alerts.append(a)

    def clear(self) -> None:
        with self._lock:
            self._alerts.clear()

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._alerts)

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            n = len(self._alerts)
            start = max(0, n - limit)
            return list(self._alerts)[start:]


__all__ = ["AlertStore", "ANOMALY_THRESHOLD"]
