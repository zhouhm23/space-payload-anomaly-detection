"""Alert service.

Routes space-side confirmed alerts (score > threshold on measured data)
into the ``AlertStore``.  This is the *measured-report* stream —
distinct from the forecast-derived *early-warning* stream managed
by ``warning_service.py``.
"""

from __future__ import annotations

from ..config import ANOMALY_THRESHOLD
from ..database.alert_store import AlertStore
from ..database.sqlite_store import SQLiteStore


class AlertService:
    def __init__(self, alerts: AlertStore, sqlite: SQLiteStore | None = None) -> None:
        self.alerts = alerts
        self.sqlite = sqlite

    def add(self, alert: dict) -> None:
        """Add an alert to both the in-memory store and SQLite."""
        self.alerts.add(alert)
        if self.sqlite is not None:
            self.sqlite.enqueue_alert(alert)

    def extend(self, alerts: list[dict]) -> None:
        """Add multiple alerts to both stores."""
        self.alerts.extend(alerts)
        if self.sqlite is not None:
            for a in alerts:
                self.sqlite.enqueue_alert(a)

    def list(self, limit: int = 50) -> list[dict]:
        return self.alerts.recent(limit)

    def clear(self) -> None:
        self.alerts.clear()

    @property
    def threshold(self) -> float:
        return ANOMALY_THRESHOLD


__all__ = ["AlertService"]
