"""Signal-generator ↔ DAQ-card IPC protocol (localhost TCP RPC).

架构上下文
==========
M1.2 把原本同进程的 SensorSource 调用拆成了两个独立进程：

- 进程 A：信号发生器（``signal_generator.py``），持有 SensorSource 实例
- 进程 B：采集卡（``main.py``），通过本模块向信号发生器请求数据

为什么不用 ``comm.SpaceServer`` / ``GroundClient``：
- SpaceServer 是「push 缓存 + client drain」模型（采集卡攒好数据，地基来取）
- IPC 需要的是「RPC 请求-响应」模型（采集卡按需拉，信号发生器现场生成）
- 协议骨架（JSON line + ``END\\n`` + ``_NumpyEncoder``）相同，但服务端处理逻辑完全不同

协议规范
========
传输：TCP 127.0.0.1（**绝不开 0.0.0.0**——IPC 是本机的，不能让外部访问）
方向：采集卡（client）→ 信号发生器（server），单连接一次请求-响应

请求（client → server，单行 JSON + ``\\n``）::

    {"channel": "C-1", "n": 512}

响应（server → client，单行 JSON + ``END\\n``）::

    {"channel": "C-1", "n": 512,
     "raw_values": [<float>, ...],     # 长度 == n（或 < n 若源已耗尽）
     "exhausted": false,                # FileSource 非 loop 模式才会 true
     "sample_rate": 100.0}
    END

错误响应::

    {"error": "unknown channel: X", "channel": "X"}
    END

并发模型
========
信号发生器主循环**单线程顺序处理**（accept → handle → close → 下一个）。
原因：SensorSource 实例是不可重入的状态机（FileSource 的 ``_pos`` 游标、
VirtualSensorSource 的 ``_t`` 计数器）。如果同一个 channel 的两个请求
并发执行，会破坏游标一致性。

采集卡侧的 ``ThreadPoolExecutor`` 不会触发并发冲突——它给每个 channel
一个 worker，每个 worker 只连一次 IPC 发自己的请求，不同 channel 访问
的是不同的 SensorSource 实例。即使两个 worker 同时连，server 也会顺序
处理（accept 后单线程执行 _handle_conn）。
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Any, Callable

import numpy as np

# 复用 comm 模块已有的 _NumpyEncoder（ndarray / numpy 标量 → JSON）
from comm import _NumpyEncoder  # noqa: E402 — space/comm.py on sys.path

logger = logging.getLogger(__name__)

# IPC 绑定地址强制 127.0.0.1（绝不对开放——契约 C18）
IPC_HOST = "127.0.0.1"
IPC_DEFAULT_PORT = 9878

# 请求超时（秒）。IPC 本地、延迟 < 1ms，2s 足够覆盖首次 virtual:sine fit 等慢路径
IPC_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Source function signature (used by server)
# ---------------------------------------------------------------------------

# 信号发生器提供给 server 的回调签名：传 channel + n，返回 (raw_values, exhausted, sample_rate)
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
        # 强制 localhost——契约 C18 禁止对外开放 IPC
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
        # 1. 读请求（单行 JSON）
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

        # 2. 调信号发生器的 source_fn（可能抛异常）
        try:
            raw, exhausted, sample_rate = self._source_fn(channel, n)
        except KeyError:
            self._send_error(sock, channel, f"unknown channel: {channel}")
            return
        except Exception as e:
            logger.exception("source_fn failed for channel=%s", channel)
            self._send_error(sock, channel, f"source error: {e}")
            return

        # 3. 回包
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

    # 便捷封装：让调用方像调 SensorSource.read 一样
    def ping(self) -> bool:
        """Quick liveness check. Returns True if server responds (even with error)."""
        try:
            # 故意发一个非法请求，只要 server 回任何 JSON 就算活着
            self.read("", 1)
            return True
        except SignalIpcError as e:
            # 只要错误信息不是连接失败，就说明 server 活着
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
