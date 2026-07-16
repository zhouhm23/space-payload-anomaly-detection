"""Centralised PHM configuration constants.

Historically these were hard-coded module-level assignments.  They are now
backed by :class:`phm.services.system_config_service.SystemConfigService`,
which loads ``data/system_config.json`` once at first access and falls back
to documented defaults when the file is missing.

Backward compatibility is preserved via PEP 562 module-level ``__getattr__``:
``from phm.config import ANOMALY_THRESHOLD`` still works unchanged.  The
constant is resolved lazily on first access, so simply importing this module
is cheap and side-effect free.

To change a value at runtime, edit ``system_config.json`` and restart (or
call ``get_system_config().reload()`` for hot reload).  Do **not** reassign
these names — they are not real module attributes anymore.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
from typing import Any

# ── Environment-driven config (secrets — stay out of JSON) ───────────────
# LLM credentials are loaded from .env via django settings; mirror them here
# so non-Django callers (e.g. management commands, experiments) still resolve.
LLM_BASE_URL: str = _os.environ.get("OPENAI_BASE_URL", "")
LLM_API_KEY: str = _os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL: str = _os.environ.get("LLM_MODEL", "deepseek-chat")

# ── RUL paths (derived, not user-tunable) ────────────────────────────────
# The C-MAPSS dataset location is fixed relative to the repo; not something
# an operator should tune via JSON.
_RUL_HERE = _Path(__file__).resolve().parent  # src/ground/phm/
RUL_CMAPSS_DATA_DIR: str = str(
    _RUL_HERE.parent.parent.parent / "datasets" / "CMAPSSData"
)

# ── Lazy loader ──────────────────────────────────────────────────────────


def _cfg() -> Any:
    """Return the SystemConfigService singleton (imported lazily to avoid a
    circular import: system_config_service lives in phm.services, which may
    import from phm at module load)."""
    from .services.system_config_service import get_system_config
    return get_system_config()


# Map of ``CONSTANT_NAME`` → ``(section, key)`` in system_config.json.
# Every name below is resolvable via module-level __getattr__ (PEP 562).
_CONSTANTS: dict[str, tuple[str, str]] = {
    # Thresholds
    "ANOMALY_THRESHOLD": ("thresholds", "anomaly"),
    "L1_CONSTANT_STD": ("thresholds", "l1_constant_std"),
    "L1_SIGMA_K": ("thresholds", "l1_sigma_k"),
    "L1_IQR_FACTOR": ("thresholds", "l1_iqr_factor"),
    "L3_CONSTANT_STD": ("thresholds", "l3_constant_std"),
    "L3_RANGE_BOOST": ("thresholds", "l3_range_boost"),
    "L3_RATE_BOOST": ("thresholds", "l3_rate_boost"),
    # Forecast
    "FORECAST_CONTEXT_LENGTH": ("forecast", "context_length"),
    "FORECAST_PREDICTION_LENGTH": ("forecast", "prediction_length"),
    # Ring buffer / SQLite
    "RING_BUFFER_MAX": ("storage", "ring_buffer_max"),
    "SQLITE_ENABLED": ("storage", "sqlite_enabled"),
    "SQLITE_BATCH_SIZE": ("storage", "sqlite_batch_size"),
    "SQLITE_FLUSH_INTERVAL": ("storage", "sqlite_flush_interval_sec"),
    # Warning lifecycle
    "WARNING_MIN_PREDICT_SCORES": ("warning", "min_predict_scores"),
    # LLM diagnosis
    "LLM_TIMEOUT_SEC": ("llm", "timeout_sec"),
    # RUL feature flags / sizing
    "RUL_ENABLED": ("rul", "enabled"),
    "RUL_WINDOW_CYCLES": ("rul", "window_cycles"),
    "RUL_HISTORY_LEN": ("rul", "history_len"),
    "RUL_POLL_INTERVAL_SEC": ("rul", "poll_interval_sec"),
}


def __getattr__(name: str) -> Any:
    """PEP 562: resolve legacy constant names from SystemConfigService.

    Falls back to the documented default if the JSON omits the key.
    Raises ``AttributeError`` for genuinely unknown names so that typos
    surface immediately (instead of silently returning None).
    """
    mapping = _CONSTANTS.get(name)
    if mapping is None:
        raise AttributeError(f"module 'phm.config' has no attribute {name!r}")
    section, key = mapping
    return _cfg().get(section, key)


# Expose the constant names for introspection (``dir(phm.config)``) and for
# tools that enumerate module attributes.
__all__ = [
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
    "RUL_CMAPSS_DATA_DIR",
    *_CONSTANTS.keys(),
]
