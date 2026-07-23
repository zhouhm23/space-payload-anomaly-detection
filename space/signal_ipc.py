"""Signal-generator ↔ DAQ-card IPC protocol (localhost TCP RPC).

Architectural context
=====================
M1.2 splits the originally in-process SensorSource calls into two separate
processes:

- Process A: the signal generator (``signal_generator.py``), holding the
  SensorSource instances.
- Process B: the DAQ card (``main.py``), which requests data from the signal
  generator through this module.

Why not ``comm.SpaceServer`` / ``GroundClient``:
- SpaceServer is a "push buffer + client drain" model (the DAQ card buffers
  data, the ground comes to fetch it).
- IPC needs an "RPC request-response" model (the DAQ card pulls on demand,
  the signal generator produces on the fly).
- The protocol skeleton (JSON line + ``END\\n`` + ``_NumpyEncoder``) is shared,
  but the server-side handling logic is entirely different.

Protocol spec
=============
Transport: TCP 127.0.0.1 only (binding the wildcard address is forbidden —
IPC is local and must not be reachable from outside).
Direction: DAQ card (client) → signal generator (server); a single connection
does one request-response.

Request (client → server, single JSON line + ``\\n``)::

    {"channel": "C-1", "n": 512}

Response (server → client, single JSON line + ``END\\n``)::

    {"channel": "C-1", "n": 512,
     "raw_values": [<float>, ...],     # length == n (or < n if the source is exhausted)
     "exhausted": false,                # only true for a non-looping FileSource
     "sample_rate": 100.0}
    END

Error response::

    {"error": "unknown channel: X", "channel": "X"}
    END

Concurrency model
=================
The signal-generator main loop is **single-threaded sequential** (accept →
handle → close → next). Reason: SensorSource instances are non-reentrant state
machines (FileSource's ``_pos`` cursor, VirtualSensorSource's ``_t`` counter).
Two concurrent requests on the same channel would corrupt the cursor.

The DAQ-card-side ``ThreadPoolExecutor`` does not trigger a concurrency clash
— it gives each channel one worker, each worker opens a single IPC connection
for its own request, and different channels touch different SensorSource
instances. Even if two workers connect simultaneously, the server still
handles them sequentially (accept followed by a single-threaded _handle_conn).
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Any, Callable

import numpy as np

# Reuse comm's existing _NumpyEncoder (ndarray / numpy scalars → JSON)
from comm import _NumpyEncoder  # noqa: E402 — space/comm.py on sys.path

logger = logging.getLogger(__name__)

# IPC bind address is forced to 127.0.0.1 (never exposed externally — contract C18)
IPC_HOST = "127.0.0.1"
IPC_DEFAULT_PORT = 9878

# Request timeout (seconds). IPC is local with < 1ms latency; 2s is plenty to
# cover slow paths like the first virtual:sine fit.
IPC_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Source function signature (used by server)
# ---------------------------------------------------------------------------

# Callback signature the signal generator hands to the server: takes channel + n,
# returns (raw_values, exhausted, sample_rate).
SourceFn = Callable[[str, int], tuple[np.ndarray, bool, float]]


# ---------------------------------------------------------------------------
# Server (runs inside signal_generator.py)
# ---------------------------------------------------------------------------

class SignalIpcServer:
    """Localhost TCP RPC server: answers DAQ read requests.

    Single-threaded accept loop — one connection at a time. Each connection
    sends one request, gets one response, then closes.

    Args:
        source_fn: callback ``(channel_name, n) -> (raw, exhausted, sample_rate)``
        port: TCP port to listen on (bound to 127.0.0.1).
    """

    def __init__(self, source_fn: SourceFn, port: int = IPC_DEFAULT_PORT):
        # Force localhost — contract C18 forbids exposing IPC externally
        self.host = IPC_HOST
        self.port = port
        self._source_fn = source_fn
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(5)
        self._sock.settimeout(1.0)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="SignalIpcServer")
        self._thread.start()
        logger.info("Signal IPC server listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        self._running.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                client, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle_conn(client)
            except Exception:
                logger.debug("IPC serve error", exc_info=True)
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    def _handle_conn(self, sock: socket.socket) -> None:
        """Read one request line → call source_fn → write one response + END."""
        # 1. Read the request (single JSON line)
        sock.settimeout(IPC_TIMEOUT)
        buf = b""
        while b"\n" not in buf and len(buf) < 8192:
            chunk = sock.recv(1)
            if not chunk:
                return  # client hung up without sending
            buf += chunk
        line = buf.decode("utf-8", errors="replace").strip()
        if not line:
            self._send_error(sock, "", "empty request")
            return
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            self._send_error(sock, "", f"invalid json: {e}")
            return

        channel = req.get("channel")
        n = req.get("n")
        if not channel or not isinstance(n, int) or n <= 0:
            self._send_error(sock, str(channel or ""),
                             "request must have {channel:str, n:int>0}")
            return

        # 2. Call the signal generator's source_fn (may raise)
        try:
            raw, exhausted, sample_rate = self._source_fn(channel, n)
        except KeyError:
            self._send_error(sock, channel, f"unknown channel: {channel}")
            return
        except Exception as e:
            logger.exception("source_fn failed for channel=%s", channel)
            self._send_error(sock, channel, f"source error: {e}")
            return

        # 3. Send the response
        resp = {
            "channel": channel,
            "n": int(len(raw)),
            "raw_values": raw.astype(np.float32, copy=False),
            "exhausted": bool(exhausted),
            "sample_rate": float(sample_rate),
        }
        payload = json.dumps(resp, cls=_NumpyEncoder, ensure_ascii=False) + "\n"
        sock.sendall(payload.encode("utf-8"))
        sock.sendall(b"END\n")

    @staticmethod
    def _send_error(sock: socket.socket, channel: str, msg: str) -> None:
        resp = {"error": msg, "channel": channel}
        payload = json.dumps(resp, ensure_ascii=False) + "\n"
        try:
            sock.sendall(payload.encode("utf-8"))
            sock.sendall(b"END\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Client (runs inside main.py / DAQ card)
# ---------------------------------------------------------------------------

class SignalIpcError(RuntimeError):
    """Raised when the IPC server returns an ``error`` response."""


class SignalIpcClient:
    """Localhost TCP RPC client: asks the signal generator for raw data.

    Each call opens a fresh connection, sends one request, reads one
    response, closes. Connections are not pooled — IPC is local, the
    connect overhead is negligible, and stateless connections are
    easier to reason about (no half-open socket after a crash).
    """

    def __init__(self, port: int = IPC_DEFAULT_PORT, timeout: float = IPC_TIMEOUT):
        self.host = IPC_HOST
        self.port = port
        self.timeout = timeout

    def read(self, channel: str, n: int) -> tuple[np.ndarray, bool, float]:
        """Request ``n`` samples for ``channel``.

        Returns ``(raw_values, exhausted, sample_rate)``.
        Raises :class:`SignalIpcError` if the server returns an error response
        or fails to respond.
        """
        req = {"channel": channel, "n": int(n)}
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            sock.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))

            # Recv until END marker
            buf = b""
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                buf += chunk
                if b"\nEND\n" in buf:
                    break
        except (socket.timeout, ConnectionError, OSError) as e:
            raise SignalIpcError(
                f"IPC read failed (channel={channel}, n={n}, "
                f"{self.host}:{self.port}): {e}"
            ) from e
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        # Parse — only the first non-empty line is the JSON response.
        text = buf.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line == "END":
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SignalIpcError(
                    f"IPC response not JSON (channel={channel}): {line!r} ({e})"
                ) from e
            if "error" in obj:
                raise SignalIpcError(
                    f"IPC server error for channel={obj.get('channel')}: {obj['error']}"
                )
            raw_values = np.array(obj["raw_values"], dtype=np.float32)
            exhausted = bool(obj.get("exhausted", False))
            sample_rate = float(obj.get("sample_rate", 0.0))
            return raw_values, exhausted, sample_rate

        raise SignalIpcError(f"IPC response had no JSON line (channel={channel}): {text!r}")

    # Convenience wrapper: lets callers use it like SensorSource.read
    def ping(self) -> bool:
        """Quick liveness check. Returns True if server responds (even with error)."""
        try:
            # Deliberately send an invalid request; any JSON reply means the server is alive
            self.read("", 1)
            return True
        except SignalIpcError as e:
            # As long as the error is not a connection failure, the server is alive
            return "IPC read failed" not in str(e)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_server(port: int = IPC_DEFAULT_PORT, timeout: float = 10.0) -> bool:
    """Block until a SignalIpcServer is accepting connections on ``port``.

    Used by main.py at startup to avoid racing the signal_generator subprocess.
    Returns True if server became available, False on timeout.
    """
    import time as _time
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        try:
            with socket.create_connection((IPC_HOST, port), timeout=0.5):
                return True
        except (ConnectionError, OSError, socket.timeout):
            _time.sleep(0.1)
    return False
