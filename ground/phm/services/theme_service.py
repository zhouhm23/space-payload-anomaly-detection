"""Front-end theme / display configuration service.

Loads ``data/ui_theme.json`` once and exposes it to the Django template
layer via a context processor (synchronous injection into ``window.THEME``).
This is the front-end counterpart to :class:`SystemConfigService` — same
load-once-with-fallback pattern, different consumer (browser vs. services).

Why a service (not just a static JSON read)?
  * Centralises the default fallbacks so the front-end never breaks on a
    malformed or missing theme file.
  * ``_strip_docs`` keeps ``_doc`` annotation keys out of the payload sent
    to the browser (they are author notes, not runtime data).
  * Future hot-reload / per-user themes can hook in here.
"""

from __future__ import annotations

import json
import logging
import os
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ThemeService", "get_theme", "reset_theme"]


# Default location: src/ground/data/ui_theme.json
_DEFAULT_THEME_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "ui_theme.json",
)

# Fallback defaults — the exact values monitor.js used to hard-code. Used
# when the JSON is missing or malformed so the page always renders.
_DEFAULTS: dict[str, Any] = {
    "colors": {
        "bgPrimary": "#0b0f1a", "bgSecondary": "#131825", "bgCard": "#1a1f2e",
        "border": "#2a3348", "textPri": "#e0e6f0", "textSec": "#8e9bb5",
        "blue": "#2d8cf0", "green": "#19be6b", "yellow": "#f5a623",
        "red": "#ed3f14", "cyan": "#00c9db",
    },
    "thresholds": {
        "anomalyScoreRed": 0.5, "anomalyScoreYellow": 0.25,
        "healthRed": 60, "healthYellow": 80,
        "rulGreen": 0.6, "rulYellow": 0.25,
    },
    "poll": {
        "chart": 2000, "health": 3000, "sensors": 3000,
        "alerts": 3000, "warnings": 3000, "rul": 5000,
        "dbStats": 5000, "diagnosis": 2000,
    },
    "chart": {
        "cacheCount": 2048, "viewCount": 512, "prefetchThreshold": 256,
        "topRatio": 0.7,
        "padding": {"top": 20, "right": 50, "bottom": 30, "left": 60},
        "gapWidthPx": 40,
    },
    "display": {
        "systemTitle": "空间站有效载荷预测性维护支持系统",
        "clockTimezone": "Asia/Shanghai",
        "datetimeFormat": "YYYY-MM-DD HH:MM:SS UTC",
    },
    "layout": {
        "headerHeight": 60, "leftPanelWidth": 240,
        "rightPanelWidth": 340, "bottomPanelFlex": 1.4,
    },
    "network": {"linkFailThreshold": 3},
}

# Keys that are documentation-only and must not reach the browser.
_DOC_KEYS = {"_doc"}


def _strip_docs(obj: Any) -> Any:
    """Recursively remove ``_doc`` keys from a nested dict/list structure."""
    if isinstance(obj, dict):
        return {k: _strip_docs(v) for k, v in obj.items() if k not in _DOC_KEYS}
    if isinstance(obj, list):
        return [_strip_docs(item) for item in obj]
    return obj


class ThemeService:
    """Load-once reader for ``ui_theme.json`` with built-in fallbacks."""

    def __init__(self, theme_path: str | None = None) -> None:
        self.theme_path = theme_path or _DEFAULT_THEME_PATH
        self._theme: dict[str, Any] = {}
        self._lock = Lock()
        self.load()

    def load(self) -> None:
        """(Re)load the JSON. Missing/malformed → fall back to defaults."""
        with self._lock:
            # Deep-copy defaults so mutation can't leak back.
            self._theme = json.loads(json.dumps(_DEFAULTS))
        if not os.path.exists(self.theme_path):
            logger.debug(
                "ui_theme.json not found at %s — using built-in defaults",
                self.theme_path,
            )
            return
        try:
            with open(self.theme_path, encoding="utf-8") as f:
                raw = json.load(f)
            with self._lock:
                # Merge: JSON overrides defaults section by section.
                for section, values in raw.items():
                    if section in _DOC_KEYS or not isinstance(values, dict):
                        continue
                    base = self._theme.setdefault(section, {})
                    base.update({
                        k: v for k, v in values.items() if k not in _DOC_KEYS
                    })
            logger.info("loaded ui_theme from %s", self.theme_path)
        except Exception:
            logger.warning(
                "failed to load ui_theme %s — using defaults",
                self.theme_path, exc_info=True,
            )

    def reload(self) -> None:
        self.load()

    def as_dict(self) -> dict[str, Any]:
        """Return the theme with ``_doc`` keys stripped (for browser injection)."""
        with self._lock:
            return _strip_docs(self._theme)


# ── Process-wide singleton ─────────────────────────────────────────────

_singleton: ThemeService | None = None
_singleton_lock = Lock()


def get_theme() -> ThemeService:
    """Return the process-wide :class:`ThemeService` singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ThemeService()
    return _singleton


def reset_theme() -> None:
    """Drop the singleton (test helper)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
