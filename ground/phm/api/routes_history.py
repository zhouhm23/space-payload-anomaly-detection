"""GET /api/history — query persisted telemetry from SQLite.
GET /api/detection — query three-layer cascade detection results.
GET /api/db-stats — SQLite row counts / health.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/api/history")
async def api_history(
    channel: str | None = Query(None),
    start: float | None = Query(None, description="Epoch seconds (inclusive)"),
    end: float | None = Query(None, description="Epoch seconds (inclusive)"),
    limit: int = Query(1000, ge=1, le=10000),
):
    """Query historical raw telemetry from the SQLite store."""
    c = deps.get()
    rows = c.sqlite.query_history(
        channel=channel,
        start_time=start,
        end_time=end,
        limit=limit,
    )
    return JSONResponse({"count": len(rows), "data": rows})


@router.get("/api/detection")
async def api_detection(
    channel: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """Query three-layer cascade detection results from SQLite.

    Also includes the latest in-memory cascade output (with per-layer
    scores) for the requested channel if available.
    """
    c = deps.get()
    rows = c.sqlite.query_detection(channel=channel, limit=limit)
    latest = None
    if channel is not None:
        cascade = c.warning_service.get_latest_cascade(channel)
        if cascade is not None:
            latest = cascade.to_dict(max_detail=True)
    return JSONResponse({
        "count": len(rows),
        "data": rows,
        "latest": latest,
    })


@router.get("/api/db-stats")
async def api_db_stats():
    """Return SQLite row counts and queue depth."""
    c = deps.get()
    return JSONResponse(c.sqlite.stats())
