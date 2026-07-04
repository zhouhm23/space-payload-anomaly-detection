"""Ground-segment HTTP API — bridges space TCP to browser ECharts frontend.

Replaces the Streamlit app with a thin FastAPI server that:
  1. Serves the static HTML frontend
  2. Exposes POST /api/poll — polls space TCP, returns chart-ready JSON
  3. Keeps a local ring buffer so the frontend always gets fresh data

Run:  python ground/server.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading
from pathlib import Path
from collections import deque

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Reuse existing comm module
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_HERE.parent / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from comm import GroundClient, TelemetryPacket, AlertPacket

HTML_PATH = _HERE / "空间站有效载荷预测性维护支持系统.html"
SPACE_HOST = os.environ.get("SPACE_HOST", "127.0.0.1")
SPACE_PORT = int(os.environ.get("SPACE_PORT", "9876"))

# Global ring buffer — per-channel for multi-channel support
MAX_BUFFER = 20000
ring_buffers: dict[str, list] = {}  # {channel_name: [{raw, score, received_at}]}
buffer_lock = threading.Lock()

app = FastAPI(title="Space Payload Health Monitor API")


def poll_space(source_id: str = "file:NASA-MSL/C-1",
               sample_rate: float = 50.0) -> tuple[dict, list, bool]:
    """Poll space TCP, return ({channel: [telemetry_entries]}, alerts, exhausted)."""
    exhausted = False
    try:
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=2)
        packets = client.poll({
            "source_id": source_id,
            "sample_rate": sample_rate,
            "use_detection": True,
        })
    except Exception:
        return {}, [], False

    channel_entries: dict[str, list] = {}
    alerts_list = []
    pkt_time = time.time()
    pkt_sr = sample_rate if sample_rate > 0 else 1.0

    for p in packets:
        if isinstance(p, TelemetryPacket):
            ch = p.channel
            raw = p.raw_values
            scores = p.scores
            n = len(raw)
            entries = channel_entries.setdefault(ch, [])
            for i in range(n):
                sample_time = pkt_time - (n - 1 - i) / pkt_sr
                entries.append({
                    "raw": float(raw[i]),
                    "score": float(scores[i]) if scores is not None and i < len(scores) else None,
                    "received_at": sample_time,
                    "channel": ch,
                })
            if p.metadata.get("exhausted", False):
                exhausted = True
        elif isinstance(p, AlertPacket):
            alerts_list.append({
                "channel": p.channel,
                "score": p.score,
                "step": p.step,
                "message": p.message,
                "time": pkt_time,
            })

    return channel_entries, alerts_list, exhausted


@app.get("/", response_class=HTMLResponse)
async def index():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>HTML not found</h1>", status_code=404)


@app.post("/api/poll")
async def api_poll(request: Request):
    """Poll space TCP, return one block of data.

    Request JSON: {source_id, sample_rate, block_size}
    Response: {channels: {ch_name: {telemetry, scores}}, alerts, exhausted, total, block_size}
    """
    body = await request.json()
    source_id = body.get("source_id", "file:NASA-MSL/C-1")
    sample_rate = float(body.get("sample_rate", 50.0))
    block_size = int(body.get("block_size", 512))

    ch_entries, alerts_list, is_exhausted = poll_space(source_id, sample_rate)

    with buffer_lock:
        for ch, entries in ch_entries.items():
            buf = ring_buffers.setdefault(ch, [])
            buf.extend(entries)
            if len(buf) > MAX_BUFFER:
                ring_buffers[ch] = buf[-MAX_BUFFER:]

        # 按 block_size 切片返回最新一块
        channels = {}
        total = 0
        for ch, buf in ring_buffers.items():
            # 取最后 block_size 个点作为当前块
            slice_buf = buf[-block_size:] if len(buf) > block_size else buf
            tele = [[int(e["received_at"] * 1000), e["raw"]] for e in slice_buf]
            sc = [[int(e["received_at"] * 1000),
                   e["score"] if e["score"] is not None else 0.0] for e in slice_buf]
            channels[ch] = {"telemetry": tele, "scores": sc}
            total += len(buf)

    return JSONResponse({
        "channels": channels,
        "alerts": alerts_list,
        "exhausted": is_exhausted,
        "total": total,
        "block_size": block_size,
    })


@app.post("/api/reset")
async def api_reset():
    global ring_buffers
    with buffer_lock:
        ring_buffers = {}
    return JSONResponse({"status": "ok"})


# ---- TTM-R3 Forecast endpoint ----
_forecaster = None

def _get_forecaster():
    global _forecaster
    if _forecaster is None:
        try:
            from forecasting import TrendForecaster
            _forecaster = TrendForecaster(device="cpu")
        except Exception as e:
            logger.warning("Failed to load TTM-R3 forecaster: %s", e)
            return None
    return _forecaster


@app.post("/api/forecast")
async def api_forecast(request: Request):
    """Run prediction on telemetry data.

    Tries TTM-R3 first; falls back to linear extrapolation if model unavailable.

    Request JSON: { "values": [float, ...] }
    Response: { "context": [float, ...], "prediction": [float, ...] }
    """
    body = await request.json()
    values = body.get("values", [])
    if len(values) < 10:
        return JSONResponse({"error": "Need at least 10 data points"}, status_code=400)

    import numpy as np
    arr = np.array(values, dtype=np.float32)

    # Try TTM-R3 first
    forecaster = _get_forecaster()
    if forecaster is not None:
        try:
            context, prediction, _ = forecaster.forecast(arr)
            return JSONResponse({
                "context": context.tolist(),
                "prediction": prediction.tolist(),
                "model": "ttm-r3",
            })
        except Exception as e:
            logger.warning("TTM-R3 forecast failed, falling back to linear: %s", e)

    # Fallback: linear extrapolation
    n = min(96, len(arr))
    recent = arr[-n:]
    x = np.arange(n, dtype=np.float64)
    y = recent.astype(np.float64)
    slope = np.polyfit(x, y, 1)[0]
    last_val = float(arr[-1])
    prediction = [last_val + slope * (i + 1) for i in range(96)]
    context = arr[-min(512, len(arr)):].tolist()

    return JSONResponse({
        "context": context,
        "prediction": prediction,
        "model": "linear",
    })


# ---- Device tree config persistence ----
CONFIG_PATH = _HERE / "device_config.json"


@app.get("/api/config")
async def api_get_config():
    """Return current device tree configuration."""
    if CONFIG_PATH.exists():
        return JSONResponse(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return JSONResponse({"device_tree": []})


@app.post("/api/config")
async def api_save_config(request: Request):
    """Save device tree configuration to disk AND push to space TCP."""
    body = await request.json()
    CONFIG_PATH.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")

    # Push tree to space segment via TCP
    try:
        client = GroundClient(host=SPACE_HOST, port=SPACE_PORT, timeout=2)
        client.poll({"device_tree": body.get("device_tree", [])})
    except Exception:
        pass

    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
