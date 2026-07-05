"""GET/POST /api/config — device-tree persistence."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/api/config")
async def api_get_config():
    c = deps.get()
    return JSONResponse(c.config.load())


@router.post("/api/config")
async def api_save_config(request: Request):
    body = await request.json()
    c = deps.get()
    return JSONResponse(c.config.save(body))
