"""Space-to-ground TCP communication.

SpaceServer — runs inside the space-segment CLI.  Accepts TCP connections
from ground clients and sends buffered telemetry/alert packets as
newline-delimited JSON.

GroundClient — runs inside the ground-segment Streamlit app.  Connects to
the space server, polls for new data, parses packets.

Protocol (plain text over TCP):
  Client connects → Server sends one JSON object per line → "END\n" → closes.

Each JSON line is one of:
  {"type":"telemetry","channel":"...","raw_values":[...],"scores":[...],...}
  {"type":"alert","channel":"...","score":0.5,"step":0,"message":"..."}
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Packet types (shared contract)
# ---------------------------------------------------------------------------

@dataclass
class TelemetryPacket:
    channel: str
    raw_values: np.ndarray
    scores: np.ndarray | None = None
    timestamp: float = field(default_factory=time.time)
    sample_rate: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertPacket:
    channel: str
    score: float
    step: int
    timestamp: float = field(default_factory=time.time)
    message: str = ""


# ---------------------------------------------------------------------------
# Space-side TCP server
# ---------------------------------------------------------------------------

class SpaceServer:
    """TCP server that runs inside the space-segment process.

    Buffers telemetry/alert packets and sends them to connecting ground
    clients on demand.  Stateless — each connection drains the buffer
    and then closes.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 max_buffer: int = 500):
        self.host = host
        self.port = port
        self._buffer: deque[dict] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    # -- public API --

    def enqueue_telemetry(self, channel: str, raw_values: np.ndarray,
                          scores: np.ndarray | None = None,
                          sample_rate: float = 1.0, **meta):
        self._enqueue({
            "type": "telemetry",
            "channel": channel,
            "raw_values": raw_values,
            "scores": scores,
            "sample_rate": sample_rate,
            "metadata": meta,
        })

    def enqueue_alert(self, channel: str, score: float, step: int,
                      message: str = ""):
        self._enqueue({
            "type": "alert",
            "channel": channel,
            "score": score,
            "step": step,
            "message": message,
        })

    def start(self):
        self._running.set()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(5)
        self._sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="SpaceServer")
        self._thread.start()
        logger.info("Space TCP server listening on %s:%d", self.host, self.port)

    def stop(self):
        self._running.clear()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    # -- internals --

    def _enqueue(self, data: dict):
        with self._lock:
            self._buffer.append(data)

    def _loop(self):
        while self._running.is_set():
            try:
                client, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._serve(client)
            except Exception:
                logger.debug("serve error", exc_info=True)
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    def _serve(self, sock: socket.socket):
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
        for item in items:
            line = json.dumps(item, cls=_NumpyEncoder, ensure_ascii=False) + "\n"
            sock.sendall(line.encode("utf-8"))
        sock.sendall(b"END\n")


# ---------------------------------------------------------------------------
# Ground-side TCP client
# ---------------------------------------------------------------------------

class GroundClient:
    """Connects to the space TCP server to fetch queued telemetry/alert packets.

    Usage::

        client = GroundClient()
        packets = client.poll()   # list[TelemetryPacket | AlertPacket]
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def poll(self, config: dict | None = None) -> list[TelemetryPacket | AlertPacket]:
        packets: list[TelemetryPacket | AlertPacket] = []
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            # Send config to space segment first
            if config:
                sock.sendall(
                    (json.dumps(config, ensure_ascii=False) + "\n").encode("utf-8")
                )
            buf = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\nEND\n" in buf:
                    break
            text = buf.decode("utf-8", errors="replace")
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line == "END":
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "telemetry":
                    packets.append(TelemetryPacket(
                        channel=obj["channel"],
                        raw_values=np.array(obj["raw_values"], dtype=np.float32),
                        scores=(np.array(obj["scores"], dtype=np.float32)
                                if obj.get("scores") else None),
                        sample_rate=obj.get("sample_rate", 1.0),
                        metadata=obj.get("metadata", {}),
                    ))
                elif obj.get("type") == "alert":
                    packets.append(AlertPacket(
                        channel=obj["channel"],
                        score=obj["score"],
                        step=obj["step"],
                        message=obj.get("message", ""),
                    ))
        except (socket.timeout, ConnectionRefusedError, OSError):
            pass
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        return packets
