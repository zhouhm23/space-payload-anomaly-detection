"""Bridge between Django and the phm service container.

Initialises the phm.api.deps Container on Django startup and manages
the auto-poll + model-eval background threads (ported from server.py).

Usage:
    services_bridge.start()          # call from AppConfig.ready()
    c = services_bridge.get_container()  # call from views
    services_bridge.stop()           # call on shutdown (optional)
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

_started = False
_auto_poll_stop = threading.Event()
_auto_poll_thread: threading.Thread | None = None
_eval_stop = threading.Event()
_eval_thread: threading.Thread | None = None

_AUTO_POLL_INTERVAL = 2.0
_AUTO_POLL_BLOCK = 512


def start() -> None:
    """Initialise phm container + start background threads. Idempotent."""
    global _started
    if _started:
        return
    from phm.api import deps
    deps.init(
        space_host=getattr(settings, 'SPACE_HOST', '127.0.0.1'),
        space_port=getattr(settings, 'SPACE_PORT', 9876),
        config_path=Path(getattr(settings, 'PHM_CONFIG_PATH', '')),
        device="cpu",
    )
    _start_auto_poll()
    _started = True
    logger.info("PHM services_bridge started (auto-poll + eval threads)")


def get_container():
    """Return the phm Container singleton.

    Lazy-initialises on first call if not yet started — this ensures the
    container exists in the request-serving process even when ready() ran
    in a different process (Django runserver spawns a child for serving).
    """
    if not _started:
        start()
    from phm.api import deps
    return deps.get()


def stop() -> None:
    """Stop background threads + flush SQLite."""
    global _started, _auto_poll_thread, _eval_thread
    if not _started:
        return
    _stop_auto_poll()
    from phm.api import deps
    deps.shutdown()
    _started = False
    logger.info("PHM services_bridge stopped")


# ── Auto-poll thread (ported from server.py:_auto_poll_loop) ────────────────

def _auto_poll_loop() -> None:
    from phm.api import deps
    while not _auto_poll_stop.is_set():
        try:
            c = deps.get()
            config_data = c.config.load()
            tree = config_data.get("device_tree", [])
            from phm.services.tree_utils import get_flat_sensors
            sources = [s.get("sourceId") for s in get_flat_sensors(tree) if s.get("sourceId")]
            if not sources:
                _auto_poll_stop.wait(_AUTO_POLL_INTERVAL)
                continue
            with ThreadPoolExecutor(max_workers=len(sources)) as pool:
                list(pool.map(lambda src: _poll_one(c, src), sources))
        except Exception:
            logger.debug("Auto-poll cycle failed", exc_info=True)
        _auto_poll_stop.wait(_AUTO_POLL_INTERVAL)


def _poll_one(c, src: str) -> None:
    try:
        c.telemetry.poll(src, 100.0, _AUTO_POLL_BLOCK)
    except Exception:
        logger.debug("Poll failed for source %s", src, exc_info=True)


# ── Model-eval thread (ported from server.py:_eval_loop) ────────────────────

def _eval_loop() -> None:
    from phm.api import deps
    while not _eval_stop.is_set():
        try:
            c = deps.get()
            for ch in c.ring.channels():
                if _eval_stop.is_set():
                    break
                try:
                    c.warning_service.evaluate_channel(ch, _AUTO_POLL_BLOCK)
                except Exception:
                    pass
        except Exception:
            logger.debug("Eval cycle failed", exc_info=True)
        _eval_stop.wait(1.0)


def _start_auto_poll() -> None:
    _auto_poll_stop.clear()
    _eval_stop.clear()
    global _auto_poll_thread, _eval_thread
    if _auto_poll_thread is None or not _auto_poll_thread.is_alive():
        _auto_poll_thread = threading.Thread(target=_auto_poll_loop, daemon=True, name="auto-poll")
        _auto_poll_thread.start()
    if _eval_thread is None or not _eval_thread.is_alive():
        _eval_thread = threading.Thread(target=_eval_loop, daemon=True, name="model-eval")
        _eval_thread.start()


def _stop_auto_poll() -> None:
    _auto_poll_stop.set()
    _eval_stop.set()
    global _auto_poll_thread, _eval_thread
    if _auto_poll_thread is not None and _auto_poll_thread.is_alive():
        _auto_poll_thread.join(timeout=5.0)
    _auto_poll_thread = None
    if _eval_thread is not None and _eval_thread.is_alive():
        _eval_thread.join(timeout=10.0)
    _eval_thread = None
