"""Ground-segment HTTP API server.

Thin entry-point: creates the FastAPI app, wires up the PHM four-layer
routers (migrated from the legacy in-file routes), and serves the Vue3
frontend build (``frontend/dist``) when available.

The business logic that used to live inline in this file has migrated to
``phm/`` (database / dataops / algorithm / services / api).  This keeps
``server.py`` to ~70 lines of glue.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# HuggingFace mirror cache (preserved from legacy server.py)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_HERE.parent / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

SPACE_HOST = os.environ.get("SPACE_HOST", "127.0.0.1")
SPACE_PORT = int(os.environ.get("SPACE_PORT", "9876"))

FRONTEND_DIST = _HERE / "frontend" / "dist"

app = FastAPI(title="Space Payload Health Monitor API")


@app.on_event("startup")
async def _startup() -> None:
    from phm.api import deps
    deps.init(
        space_host=SPACE_HOST,
        space_port=SPACE_PORT,
        config_path=_HERE / "device_config.json",
        device="cpu",
    )
    _start_auto_poll()
    from phm.api import (
        poll_router,
        forecast_router,
        config_router,
        reset_router,
        health_router,
        alerts_router,
        warnings_router,
        sensors_router,
        history_router,
        window_router,
        export_router,
    )
    for r in (
        poll_router,
        forecast_router,
        config_router,
        reset_router,
        health_router,
        alerts_router,
        warnings_router,
        sensors_router,
        history_router,
        window_router,
        export_router,
    ):
        app.include_router(r)


@app.on_event("shutdown")
async def _shutdown() -> None:
    _stop_auto_poll()
    from phm.api import deps
    deps.shutdown()
    logger.info("PHM deps shut down (SQLite flushed)")


# ---- Auto-poll background thread ------------------------------------------
# A daemon thread that periodically polls the space segment and ingests
# data into RingBuffer + SQLite, so the frontend only needs to read from
# SQLite via /api/window.  This decouples data ingestion from display.

import threading  # noqa: E402

# Poll interval: 2 s.  The space segment produces data in 512-pt blocks
# (5.12 s at 100 Hz) but serially across 4 channels, so a block for any
# given channel arrives roughly every 20-40 s.  Polling at 2 s ensures we
# drain each block as soon as it appears, rather than letting it sit in
# the space buffer and creating gaps.  Overlapping polls (polling faster
# than the block duration) are handled by _last_ts in telemetry_service
# which keeps consecutive timestamp ranges non-overlapping.
#
# Model evaluation (TTM-R3 forecast + TSPulse detection) runs in a
# SEPARATE thread so its latency does not delay polling.
_AUTO_POLL_INTERVAL = 2.0
_AUTO_POLL_BLOCK = 512

_auto_poll_stop = threading.Event()
_auto_poll_thread: threading.Thread | None = None
_eval_stop = threading.Event()
_eval_thread: threading.Thread | None = None


def _auto_poll_loop() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from phm.api import deps
    poll_count = 0
    while not _auto_poll_stop.is_set():
        try:
            c = deps.get()
            # Gather all configured sensor sources — recursively walk the
            # device tree so sensors nested inside folders are also polled.
            # (Previously only top-level sourceId was read, which silently
            # skipped foldered sensors.)
            config_data = c.config.load()
            tree = config_data.get("device_tree", [])
            from phm.services.tree_utils import get_flat_sensors
            sources = [s.get("sourceId") for s in get_flat_sensors(tree) if s.get("sourceId")]
            if not sources:
                _auto_poll_stop.wait(_AUTO_POLL_INTERVAL)
                continue

            # Poll all sources in parallel — each source is an independent
            # TCP connection to the space segment.  The ingest (ring +
            # SQLite + _last_ts update) happens inside telemetry.poll() and
            # is protected by per-resource locks (RingBuffer._lock,
            # AlertStore._lock, SQLiteStore._queue), so concurrent polls
            # from different sources are safe.
            with ThreadPoolExecutor(max_workers=len(sources)) as pool:
                list(pool.map(
                    lambda src: _poll_one(c, src),
                    sources,
                ))

            poll_count += 1
        except Exception:
            logger.debug("Auto-poll cycle failed", exc_info=True)
        _auto_poll_stop.wait(_AUTO_POLL_INTERVAL)


def _poll_one(c, src: str) -> None:
    """Poll a single source — wrapped so ThreadPoolExecutor swallows errors."""
    try:
        c.telemetry.poll(src, 100.0, _AUTO_POLL_BLOCK)
    except Exception:
        logger.debug("Poll failed for source %s", src, exc_info=True)


def _eval_loop() -> None:
    """Run model evaluation (forecast + detection) for all channels.

    Runs in its own thread so slow model inference does not delay the
    poll thread.  Iterates channels serially — PyTorch models are not
    thread-safe for concurrent forward passes on the same model object.
    """
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
        # Poll eval frequently; if models are slow the loop naturally
        # throttles to one eval-cycle per (sum of per-channel inference).
        _eval_stop.wait(1.0)


def _start_auto_poll() -> None:
    _auto_poll_stop.clear()
    _eval_stop.clear()
    global _auto_poll_thread, _eval_thread
    if _auto_poll_thread is None or not _auto_poll_thread.is_alive():
        _auto_poll_thread = threading.Thread(target=_auto_poll_loop, daemon=True, name="auto-poll")
        _auto_poll_thread.start()
        logger.info("Auto-poll thread started (interval=%.1fs)", _AUTO_POLL_INTERVAL)
    if _eval_thread is None or not _eval_thread.is_alive():
        _eval_thread = threading.Thread(target=_eval_loop, daemon=True, name="model-eval")
        _eval_thread.start()
        logger.info("Model-eval thread started")


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


# ---- Static frontend ----
# Prefer the Vue3 build; fall back to the legacy single-file HTML while the
# migration is in progress.  Once the frontend is fully migrated, the legacy
# HTML is deleted and only dist/ is served.

if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="assets",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse((FRONTEND_DIST / "index.html").read_text(encoding="utf-8"))
else:

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(
            "<h1>Frontend not built</h1><p>Run <code>npm run build</code> in frontend/</p>",
            status_code=404,
        )

# Standalone HTML frontend — no build step, no Vue, pure HTML/CSS/JS
_STANDALONE_HTML = FRONTEND_DIST.parent / "standalone.html"


@app.get("/standalone", response_class=HTMLResponse)
async def standalone():
    if _STANDALONE_HTML.exists():
        return HTMLResponse(_STANDALONE_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>standalone.html not found</h1>", status_code=404)


# New single-HTML dashboard (Day 10 rebuild). Lives alongside standalone.html
# inside the public product repo's frontend/ dir — no cross-repo dependency.
_DASHBOARD_HTML = FRONTEND_DIST.parent / "dashboard.html"


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if _DASHBOARD_HTML.exists():
        return HTMLResponse(_DASHBOARD_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard HTML not found</h1>", status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)