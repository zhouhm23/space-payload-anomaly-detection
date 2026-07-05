"""GET /api/alerts — confirmed measured anomaly alerts."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/api/alerts")
async def api_alerts(limit: int = Query(50, ge=1, le=500)):
    c = deps.get()
    return JSONResponse({
        "alerts": c.alert_service.list(limit),
        "threshold": c.alert_service.threshold,
    })
