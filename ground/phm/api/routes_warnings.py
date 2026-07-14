"""GET /api/warnings — forecast-derived early warnings (lifecycle view).
POST /api/warnings/{id}/verdict — human verdict annotation.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from . import deps
from ..database.warning_store import _VALID_VERDICTS

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


class VerdictRequest(BaseModel):
    """Body for ``POST /api/warnings/{id}/verdict``."""
    human_verdict: str

    @field_validator("human_verdict")
    @classmethod
    def _validate_verdict(cls, v: str) -> str:
        if v not in _VALID_VERDICTS:
            raise ValueError(f"human_verdict must be one of {sorted(_VALID_VERDICTS)}")
        return v


@router.post("/api/warnings/{warning_id}/verdict")
async def api_warning_verdict(warning_id: int, body: VerdictRequest):
    """Set a human verdict (real / false_alarm / uncertain) on a warning."""
    c = deps.get()
    ok = c.warning_service.warnings.set_verdict(warning_id, "human", body.human_verdict)
    if not ok:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "id": warning_id, "human_verdict": body.human_verdict})
