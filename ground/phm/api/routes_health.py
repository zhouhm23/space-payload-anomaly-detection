"""GET /api/health — per-channel + system health values."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/api/health")
async def api_health(block_size: int = Query(20000, ge=1, le=20000)):
    c = deps.get()
    return JSONResponse(c.health.system_health(block_size))
