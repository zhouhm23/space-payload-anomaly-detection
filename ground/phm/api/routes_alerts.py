"""GET /api/alerts — confirmed measured anomaly alerts (in-memory, real-time).
GET /api/alerts/history — persisted alert records from SQLite (with id/status).
PATCH /api/alerts/{alert_id} — update an alert's lifecycle status.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import deps

router = APIRouter()


@router.get("/api/alerts")
async def api_alerts(limit: int = Query(50, ge=1, le=500)):
    c = deps.get()
    return JSONResponse({
        "alerts": c.alert_service.list(limit),
        "threshold": c.alert_service.threshold,
    })


@router.get("/api/alerts/history")
async def api_alerts_history(limit: int = Query(50, ge=1, le=500)):
    """Return persisted alert records (measured + predicted) from SQLite.

    Unlike ``GET /api/alerts`` (which reads the in-memory deque and has no
    ``id`` / ``status``), this endpoint reads ``alert_records`` and returns
    each row's ``id`` so the frontend can target it with PATCH.
    """
    c = deps.get()
    rows = c.sqlite.query_alerts(limit=limit)
    return JSONResponse({
        "alerts": rows,
        "threshold": c.alert_service.threshold,
    })


class AlertStatusPatch(BaseModel):
    """Body for ``PATCH /api/alerts/{alert_id}``."""
    status: str = Field(..., description="pending | confirmed | false")


@router.patch("/api/alerts/{alert_id}")
async def api_patch_alert(alert_id: int, body: AlertStatusPatch):
    """Update an alert record's lifecycle status (e.g. mark a pending
    predicted warning as ``confirmed`` or ``false`` after manual review).
    """
    c = deps.get()
    ok = c.sqlite.update_alert_status(alert_id, body.status)
    if not ok:
        return JSONResponse(
            {"ok": False, "error": "not_found_or_invalid_status"},
            status_code=404,
        )
    return JSONResponse({"ok": True, "id": alert_id, "status": body.status})
