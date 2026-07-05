"""POST /api/poll — poll space TCP, ingest, return chart-ready JSON.

Response contract is identical to the legacy implementation so the
existing frontend (and the Vue3 rewrite) need no changes.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.post("/api/poll")
async def api_poll(request: Request):
    body = await request.json()
    source_id = body.get("source_id", "file:NASA-MSL/C-1")
    sample_rate = float(body.get("sample_rate", 50.0))
    block_size = int(body.get("block_size", 512))

    c = deps.get()
    result = c.telemetry.poll(source_id, sample_rate, block_size)

    # Run the warning evaluator for each channel that just produced data.
    # Best-effort: never let warning failures break the poll response.
    ingested = result.pop("_ingested", {})
    for ch in ingested:
        try:
            c.warning_service.evaluate_channel(ch, block_size)
        except Exception:
            pass

    return JSONResponse(result)
