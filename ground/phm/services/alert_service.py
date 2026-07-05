"""Alert service.

Routes space-side confirmed alerts (score > threshold on measured data)
into the ``AlertStore``.  This is the *measured-report* (实报) stream —
distinct from the forecast-derived *early-warning* (预警) stream managed
by ``warning_service.py``.
"""

from __future__ import annotations

from ..config import ANOMALY_THRESHOLD
from ..database.alert_store import AlertStore


class AlertService:
    def __init__(self, alerts: AlertStore) -> None:
        self.alerts = alerts

    def list(self, limit: int = 50) -> list[dict]:
        return self.alerts.recent(limit)

    def clear(self) -> None:
        self.alerts.clear()

    @property
    def threshold(self) -> float:
        return ANOMALY_THRESHOLD


__all__ = ["AlertService"]
