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

    Scans ``src/ground/models/rul/`` for weight+scaler pairs and the C-MAPSS
    data directory for matching test files.  Each subset with all three
    assets (``fd00X_lstm_attn.pt`` + ``scaler_fd00X.json`` +
    ``test_FD00X.txt``) gets its own :class:`CMAPSSDataSource` and
    :class:`RULPredictor`.  Subsets with incomplete assets are skipped with
    a warning.  If none qualify, the service stays disabled (503).

    This is data-driven — adding a new subset (e.g. FD002 once its scaler
    is trained) only requires dropping the files in place; no code edit.
    """
    if not RUL_ENABLED:
        logger.info("RUL service disabled by config (RUL_ENABLED=False)")
        return None

    from ..algorithm import RULPredictor  # local import keeps startup lean

    data_dir = Path(RUL_CMAPSS_DATA_DIR)
    models_dir = Path(__file__).resolve().parent.parent.parent / "models" / "rul"

    # Discover every subset that has both weights and a scaler JSON.
    # Pattern: scaler_fd00X.json + fd00X_lstm_attn.pt + test_FD00X.txt
    available_subsets: list[str] = []
    for scaler_file in sorted(models_dir.glob("scaler_fd00*.json")):
        # scaler_fd001.json → "fd001"
        subset_lower = scaler_file.stem.replace("scaler_", "")
        weights_file = models_dir / f"{subset_lower}_lstm_attn.pt"
        subset_upper = subset_lower.upper()
        test_file = data_dir / f"test_{subset_upper}.txt"
        if weights_file.exists() and test_file.exists():
            available_subsets.append(subset_upper)
        else:
            logger.warning(
                "RUL subset %s skipped — scaler found but weights (%s) "
                "or test data (%s) missing.",
                subset_upper, weights_file, test_file,
            )

    if not available_subsets:
        logger.warning(
            "RUL service disabled — no complete subset assets under %s. "
            "Each subset needs scaler_fd00X.json + fd00X_lstm_attn.pt + "
            "test_FD00X.txt.",
            models_dir,
        )
        return None

    import numpy as np
    predictors: dict[str, RULPredictor] = {}
    data_sources: dict[str, CMAPSSDataSource] = {}
    try:
        for subset in available_subsets:
            tag = subset.lower()
            data_sources[tag] = CMAPSSDataSource(data_dir, subset=subset)
            predictors[tag] = RULPredictor(subset=subset)
            # Eagerly pre-warm the model NOW (during init, before the
            # auto-poll/eval threads start spinning up TSPulse).  Lazy
            # loading defers LSTM creation to the first /api/rul call,
            # where a concurrent tsfm_public from_pretrained can leave
            # torch in a meta-tensor state and corrupt the new module.
            predictors[tag].predict_rul(
                np.zeros(
                    (RUL_WINDOW_CYCLES, len(CMAPSSDataSource.SENSORS)),
                    dtype=np.float32,
                ),
                raw=True,
            )
        rul = RulService(
            data_sources=data_sources,
            predictors=predictors,
            config_service=config,
            window_cycles=RUL_WINDOW_CYCLES,
            history_len=RUL_HISTORY_LEN,
        )
        logger.info(
            "RUL service enabled (subsets=%s, window=%d cycles, models pre-warmed)",
            available_subsets, RUL_WINDOW_CYCLES,
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
