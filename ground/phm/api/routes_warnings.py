"""GET /api/warnings — forecast-derived early warnings (lifecycle view)."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/api/warnings")
async def api_warnings(limit: int = Query(50, ge=1, le=500)):
    c = deps.get()
    return JSONResponse({"warnings": c.warning_service.list(limit)})


@router.get("/api/predict-scores")
async def api_predict_scores(channel: str = Query(...)):
    c = deps.get()
    data = c.warning_service.get_latest_predict_scores(channel)
    if data is None:
        return JSONResponse({"timestamps": [], "scores": [], "predict_start": 0, "predict_end": 0})
    return JSONResponse(data)
