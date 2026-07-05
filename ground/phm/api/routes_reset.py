"""POST /api/reset — clear in-memory ring buffer + alert/warning stores."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.post("/api/reset")
async def api_reset():
    c = deps.get()
    c.ring.clear()
    c.alerts.clear()
    c.warnings.clear()
    return JSONResponse({"status": "ok"})
