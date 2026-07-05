"""POST /api/forecast — TTM-R3 prediction (with linear fallback)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.post("/api/forecast")
async def api_forecast(request: Request):
    body = await request.json()
    values = body.get("values", [])
    c = deps.get()
    result = c.forecast.forecast(values)
    status = 400 if "error" in result else 200
    return JSONResponse(result, status_code=status)
