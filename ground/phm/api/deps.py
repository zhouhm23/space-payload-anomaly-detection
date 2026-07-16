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
    RUL_ENABLED, RUL_CMAPSS_DATA_DIR, RUL_WINDOW_CYCLES, RUL_HISTORY_LEN,
)
from ..database import RingBuffer, SQLiteStore
from ..database.alert_store import AlertStore
from ..database.warning_store import WarningStore
from ..services.alert_service import AlertService
from ..services.config_service import ConfigService
from ..services.diagnosis_service import DiagnosisService
from ..services.forecast_service import ForecastService
from ..services.health_service import HealthService
from ..services.rul_service import CMAPSSDataSource, RulService
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
    rul: RulService | None = None  # None when data/weights missing (503)


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

    # RUL service — optional.  Built only when the C-MAPSS data dir and at
    # least the FD001 weights + scaler are present.  Failure is non-fatal:
    # c.rul stays None and /api/rul returns 503.
    rul = _maybe_build_rul(config)

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
        rul=rul,
    )
    return _container


def _maybe_build_rul(config: ConfigService) -> RulService | None:
    """Construct the RUL service if assets are present; else None.

    Looks for the C-MAPSS data directory (test_FD001.txt) and the FD001
    model weights + scaler JSON under ``src/ground/models/rul/``.  Any
    missing piece → return None (logged) so the rest of the stack starts.
    """
    if not RUL_ENABLED:
        logger.info("RUL service disabled by config (RUL_ENABLED=False)")
        return None

    from ..algorithm import RULPredictor  # local import keeps startup lean

    data_dir = Path(RUL_CMAPSS_DATA_DIR)
    if not (data_dir / "test_FD001.txt").exists():
        logger.warning(
            "RUL service disabled — C-MAPSS data not found at %s. "
            "Place test_FD001.txt/train_FD001.txt/RUL_FD001.txt there.",
            data_dir,
        )
        return None

    models_dir = Path(__file__).resolve().parent.parent.parent / "models" / "rul"
    fd001_weights = models_dir / "fd001_lstm_attn.pt"
    fd001_scaler = models_dir / "scaler_fd001.json"
    if not (fd001_weights.exists() and fd001_scaler.exists()):
        logger.warning(
            "RUL service disabled — FD001 weights (%s) or scaler (%s) missing.",
            fd001_weights, fd001_scaler,
        )
        return None

    try:
        data_source = CMAPSSDataSource(data_dir, subset="FD001")
        predictors = {"fd001": RULPredictor(subset="FD001")}
        # Eagerly load the model NOW (during init, before the auto-poll/eval
        # threads start spinning up TSPulse).  Lazy loading defers the LSTM
        # creation to the first /api/rul call, where a concurrent tsfm_public
        # from_pretrained can leave torch in a meta-tensor state and corrupt
        # the new module.  Loading here avoids that race entirely.
        import numpy as np
        predictors["fd001"].predict_rul(
            np.zeros((RUL_WINDOW_CYCLES, 14), dtype=np.float32), raw=True
        )
        rul = RulService(
            data_source=data_source,
            predictors=predictors,
            config_service=config,
            window_cycles=RUL_WINDOW_CYCLES,
            history_len=RUL_HISTORY_LEN,
        )
        logger.info(
            "RUL service enabled (FD001, %d engines, window=%d cycles, model pre-warmed)",
            len(data_source.channels()), RUL_WINDOW_CYCLES,
        )
        return rul
    except Exception as e:  # never let RUL construction abort startup
        logger.exception("RUL service construction failed: %s", e)
        return None


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
