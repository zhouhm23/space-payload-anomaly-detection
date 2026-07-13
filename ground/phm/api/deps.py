"""Shared dependency container.

Holds the singletons (RingBuffer, AlertStore, WarningStore, SQLiteStore,
CascadeDetector, services) so that every route touches the same in-memory
+ persistent state.  ``init()`` is called once from ``server.py`` at
startup with runtime parameters (host/port, config path).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    SQLITE_ENABLED, SQLITE_BATCH_SIZE, SQLITE_FLUSH_INTERVAL,
    L1_CONSTANT_STD, L1_SIGMA_K, L1_IQR_FACTOR,
    L3_CONSTANT_STD, L3_RANGE_BOOST, L3_RATE_BOOST,
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
)
from ..database import RingBuffer, SQLiteStore
from ..database.alert_store import AlertStore
from ..database.warning_store import WarningStore
from ..services.alert_service import AlertService
from ..services.config_service import ConfigService
from ..services.diagnosis_service import DiagnosisService
from ..services.forecast_service import ForecastService
from ..services.health_service import HealthService
from ..services.telemetry_service import TelemetryService
from ..services.warning_service import WarningService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    ring: RingBuffer
    alerts: AlertStore
    warnings: WarningStore
    sqlite: SQLiteStore
    telemetry: TelemetryService
    forecast: ForecastService
    health: HealthService
    alert_service: AlertService
    warning_service: WarningService
    config: ConfigService
    diagnosis: DiagnosisService


_container: Container | None = None


def init(
    *,
    space_host: str = "127.0.0.1",
    space_port: int = 9876,
    config_path: Path | None = None,
    device: str = "cpu",
    db_path: Path | None = None,
) -> Container:
    global _container
    ring = RingBuffer()
    alerts = AlertStore()
    warnings = WarningStore()

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "device_config.json"

    if db_path is None:
        db_path = Path(__file__).resolve().parent.parent.parent / "data" / "phm.db"

    sqlite = SQLiteStore(
        db_path,
        batch_size=SQLITE_BATCH_SIZE,
        flush_interval=SQLITE_FLUSH_INTERVAL,
        enabled=SQLITE_ENABLED,
    )
    sqlite.start()

    forecast = ForecastService(device=device)
    telemetry = TelemetryService(ring, alerts, sqlite, space_host, space_port)
    # ConfigService is built before HealthService so the latter can be wired
    # with the tree (for folder-level health aggregation).  ConfigService has
    # no upstream deps, so moving it earlier keeps the graph acyclic.
    config = ConfigService(config_path, space_host, space_port)
    health = HealthService(ring, config)
    alert_service = AlertService(alerts, sqlite)
    warning_service = WarningService(ring, warnings, forecast, sqlite)
    diagnosis = DiagnosisService(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        warning_service=warning_service,
        sqlite_store=sqlite,
        config_service=config,
    )
    if diagnosis.enabled:
        logger.info(
            "LLM diagnosis enabled (model=%s, base=%s)",
            LLM_MODEL, LLM_BASE_URL,
        )
    else:
        logger.info("LLM diagnosis disabled — set OPENAI_API_KEY/OPENAI_BASE_URL/LLM_MODEL to enable")

    _container = Container(
        ring=ring,
        alerts=alerts,
        warnings=warnings,
        sqlite=sqlite,
        telemetry=telemetry,
        forecast=forecast,
        health=health,
        alert_service=alert_service,
        warning_service=warning_service,
        config=config,
        diagnosis=diagnosis,
    )
    return _container


def get() -> Container:
    if _container is None:
        raise RuntimeError("phm.api.deps not initialised — call init() first")
    return _container


def shutdown() -> None:
    """Flush and close the SQLite store (call on app shutdown)."""
    global _container
    if _container is not None:
        _container.sqlite.close()
    _container = None


__all__ = ["Container", "init", "get", "shutdown"]
