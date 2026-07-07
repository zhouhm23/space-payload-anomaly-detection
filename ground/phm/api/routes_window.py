"""GET /api/window — latest N telemetry points + predictions from SQLite.

This is the primary data source for the frontend scrolling window viewer.
The frontend only needs "give me the last *count* points for channel X"
to render the signal chart and the anomaly-score chart.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/api/window")
async def api_window(
    channel: str = Query(..., description="Channel name, e.g. C-1"),
    count: int = Query(512, ge=1, le=10000, description="Window length (points)"),
    end_ts: float | None = Query(None, description="Right-edge epoch seconds (None=latest)"),
):
    """Return the latest ``count`` raw telemetry points for *channel*.

    Also returns any prediction batches whose origin falls within the
    window, so the frontend can draw predicted-value and predicted-score
    dashed lines without a second request.
    """
    c = deps.get()
    result = c.sqlite.query_window(
        channel=channel,
        count=count,
        end_ts=end_ts,
    )
    return JSONResponse(result)
