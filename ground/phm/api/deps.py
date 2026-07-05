"""Shared dependency container.

Holds the singletons (RingBuffer, AlertStore, WarningStore, services) so
that every route touches the same in-memory state.  ``init()`` is called
once from ``server.py`` at startup with runtime parameters (host/port,
config path).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..database import RingBuffer
from ..database.alert_store import AlertStore
from ..database.warning_store import WarningStore
from ..services.alert_service import AlertService
from ..services.config_service import ConfigService
from ..services.forecast_service import ForecastService
from ..services.health_service import HealthService
from ..services.telemetry_service import TelemetryService
from ..services.warning_service import WarningService


@dataclass
class Container:
    ring: RingBuffer
    alerts: AlertStore
    warnings: WarningStore
    telemetry: TelemetryService
    forecast: ForecastService
    health: HealthService
    alert_service: AlertService
    warning_service: WarningService
    config: ConfigService


_container: Container | None = None


def init(
    *,
    space_host: str = "127.0.0.1",
    space_port: int = 9876,
    config_path: Path | None = None,
    device: str = "cpu",
) -> Container:
    global _container
    ring = RingBuffer()
    alerts = AlertStore()
    warnings = WarningStore()

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "device_config.json"

    forecast = ForecastService(device=device)
    telemetry = TelemetryService(ring, alerts, space_host, space_port)
    health = HealthService(ring)
    alert_service = AlertService(alerts)
    warning_service = WarningService(ring, warnings, forecast)
    config = ConfigService(config_path, space_host, space_port)

    _container = Container(
        ring=ring,
        alerts=alerts,
        warnings=warnings,
        telemetry=telemetry,
        forecast=forecast,
        health=health,
        alert_service=alert_service,
        warning_service=warning_service,
        config=config,
    )
    return _container


def get() -> Container:
    if _container is None:
        raise RuntimeError("phm.api.deps not initialised — call init() first")
    return _container


__all__ = ["Container", "init", "get"]
