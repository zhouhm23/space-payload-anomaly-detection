"""POST /api/diagnosis — on-demand LLM anomaly diagnosis for a channel."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import deps

router = APIRouter()


class DiagnosisRequest(BaseModel):
    channel: str = Field(..., description="Telemetry channel name, e.g. 'C-1'")
    alert_type: str = Field("measured", description="'measured' or 'predicted'")
    alert_ts: float | None = Field(None, description="Alert/warning timestamp — cache key")


@router.post("/api/diagnosis")
async def api_diagnosis(req: DiagnosisRequest):
    """Produce a Markdown diagnosis report for one channel.

    Aggregates three-layer cascade output, recent telemetry statistics,
    historical alerts, device-tree position, and offline calibration into a
    structured prompt, then calls an OpenAI-compatible LLM.  On-demand
    only — never runs automatically.  Results are cached in SQLite keyed
    by (channel, alert_type, alert_ts) — repeated clicks return instantly.
    """
    c = deps.get()
    if not c.diagnosis.enabled:
        return JSONResponse(
            status_code=503,
            content={
                "error": "LLM diagnosis not configured",
                "detail": "Set OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL environment variables.",
            },
        )
    result = c.diagnosis.diagnose(req.channel, alert_type=req.alert_type, alert_ts=req.alert_ts)
    if result.get("error"):
        # LLM call failed or no data — return 502 with the detail.
        return JSONResponse(
            status_code=502 if "LLM" in result["error"] else 404,
            content=result,
        )
    return JSONResponse(content=result)


@router.get("/api/diagnosis/done")
async def api_diagnosis_done(limit: int = Query(200, ge=1, le=1000)):
    """Return the set of (channel, alert_type, alert_ts) already diagnosed.

    Frontend uses this to mark the "诊断" button green on alerts whose
    diagnosis is cached, so the user sees at a glance which alerts have
    been analysed.
    """
    c = deps.get()
    if c.sqlite is None:
        return JSONResponse(content={"done": []})
    items = c.sqlite.list_diagnosis_keys(limit=limit)
    return JSONResponse(content={"done": items})
