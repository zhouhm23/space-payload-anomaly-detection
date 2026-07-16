"""System-wide runtime configuration service.

Loads ``data/system_config.json`` once and exposes typed accessors for every
value that ``phm/config.py`` previously hard-coded.  This is the *single
source of truth* at runtime — ``config.py`` now reads from here lazily via
module-level ``__getattr__`` (PEP 562), so existing ``from phm.config import
ANOMALY_THRESHOLD`` imports keep working without any caller change.

Design (mirrors ``CalibrationConfig``):
  * Constructor takes an optional path; falls back to the default location.
  * ``load()`` reads the JSON; a missing or malformed file logs a warning
    and falls back to ``_DEFAULTS`` (the code never crashes on bad config).
  * Each section is exposed as a property returning a dict; individual
    values via ``get(section, key)`` for ad-hoc access.
  * Agent-friendly: ``snapshot()`` returns the whole config as a dict for
    ``manage.py config`` and ``GET /api/config/system``.

The defaults embedded here are the exact values that were hard-coded in
``config.py`` before this refactor — so behaviour is identical when the JSON
is absent.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SystemConfigService", "get_system_config", "reset_system_config"]


# Default location: src/ground/data/system_config.json
# Path from here (src/ground/phm/services/): up 3 → src/ground/, then data/.
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "system_config.json",
)

# Fallback defaults — the exact values config.py used to hard-code.  Used
# when the JSON is missing or a key is absent so the system always runs.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "network": {
        "space_host": "127.0.0.1",
        "space_port": 9876,
        "ground_port": 8501,
        "link_fail_threshold": 3,
    },
    "storage": {
        "db_path": "data/phm.db",
        "ring_buffer_max": 20000,
        "sqlite_batch_size": 200,
        "sqlite_flush_interval_sec": 2.0,
        "sqlite_enabled": True,
    },
    "thresholds": {
        "anomaly": 0.5,
        "l1_constant_std": 1e-3,
        "l1_sigma_k": 3.0,
        "l1_iqr_factor": 1.5,
        "l3_constant_std": 1e-3,
        "l3_range_boost": 0.95,
        "l3_rate_boost": 0.85,
    },
    "forecast": {
        "context_length": 512,
        "prediction_length": 96,
    },
    "warning": {
        "min_predict_scores": 1,
    },
    "rul": {
        "enabled": True,
        "window_cycles": 30,
        "history_len": 20,
        "poll_interval_sec": 5.0,
    },
    "llm": {
        "timeout_sec": 30.0,
    },
}

# Keys that are documentation-only and should never be surfaced as config.
_DOC_KEYS = {"_doc"}


class SystemConfigService:
    """Typed, reloadable reader for ``system_config.json``.

    Thread-safe (a background reload could be added later; for now load()
    is called once at construction).
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._cfg: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self.load()

    def load(self) -> None:
        """(Re)load the JSON. Missing file → use ``_DEFAULTS`` entirely.
        Malformed file → log warning and keep defaults."""
        with self._lock:
            self._cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
        if not os.path.exists(self.config_path):
            logger.debug(
                "system_config.json not found at %s — using built-in defaults",
                self.config_path,
            )
            return
        try:
            with open(self.config_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Merge: JSON overrides defaults section by section, key by key.
            # Unknown sections/keys are kept (forward-compat) but _doc keys
            # are stripped from the typed view.
            with self._lock:
                for section, values in raw.items():
                    if section in _DOC_KEYS or not isinstance(values, dict):
                        continue
                    base = self._cfg.setdefault(section, {})
                    for k, v in values.items():
                        if k not in _DOC_KEYS:
                            base[k] = v
            logger.info(
                "loaded system config from %s (%d sections)",
                self.config_path, len(self._cfg),
            )
        except Exception:
            logger.warning(
                "failed to load system config %s — using defaults",
                self.config_path, exc_info=True,
            )

    def reload(self) -> None:
        """Alias for :meth:`load` (hot-reload use case)."""
        self.load()

    # ── Typed accessors (one per config section) ───────────────────────

    @property
    def network(self) -> dict[str, Any]:
        return self._cfg["network"]

    @property
    def storage(self) -> dict[str, Any]:
        return self._cfg["storage"]

    @property
    def thresholds(self) -> dict[str, Any]:
        return self._cfg["thresholds"]

    @property
    def forecast(self) -> dict[str, Any]:
        return self._cfg["forecast"]

    @property
    def warning(self) -> dict[str, Any]:
        return self._cfg["warning"]

    @property
    def rul(self) -> dict[str, Any]:
        return self._cfg["rul"]

    @property
    def llm(self) -> dict[str, Any]:
        return self._cfg["llm"]

    # ── Ad-hoc access ──────────────────────────────────────────────────

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Return ``config[section][key]``, or ``default`` if absent."""
        return self._cfg.get(section, {}).get(key, default)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the full config (for CLI / API exposure).

        Strips ``_doc`` keys — those are author annotations, not runtime data.
        """
        with self._lock:
            return {
                section: {k: v for k, v in values.items() if k not in _DOC_KEYS}
                for section, values in self._cfg.items()
            }


# ── Process-wide singleton ─────────────────────────────────────────────
# Lazily constructed on first access so importing this module is cheap.
# ``config.py``'s ``__getattr__`` calls ``get_system_config()`` on demand.

_singleton: SystemConfigService | None = None
_singleton_lock = Lock()


def get_system_config() -> SystemConfigService:
    """Return the process-wide :class:`SystemConfigService` singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = SystemConfigService(_DEFAULT_CONFIG_PATH)
    return _singleton


def reset_system_config() -> None:
    """Drop the singleton (test helper — forces re-creation on next access)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
