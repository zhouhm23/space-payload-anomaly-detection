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
    ):
        app.include_router(r)


@app.on_event("shutdown")
async def _shutdown() -> None:
    from phm.api import deps
    deps.shutdown()
    logger.info("PHM deps shut down (SQLite flushed)")


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)