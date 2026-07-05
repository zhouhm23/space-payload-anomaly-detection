"""GET /api/sensors — dashboard snapshot (latest value/score/health per channel)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/api/sensors")
async def api_sensors():
    c = deps.get()
    latest = c.ring.latest_metrics()
    health = c.health.system_health()
    sensors = []
    for ch, m in latest.items():
        sensors.append({
            "channel": ch,
            "latest_raw": m["raw"],
            "latest_score": m["score"],
            "points": m["points"],
            "received_at": m["received_at"],
            "health": health["channels"].get(ch, 100.0),
        })
    return JSONResponse({
        "sensors": sensors,
        "system_health": health["system"],
    })
