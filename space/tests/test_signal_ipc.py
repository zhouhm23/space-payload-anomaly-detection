"""Unit tests for the signal-generator ↔ DAQ-card IPC protocol.

These tests start a real SignalIpcServer on an ephemeral port (avoids
clashing with the production port 9878) and exercise the full
client → server → client round-trip over actual localhost TCP.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest

# space/ is on sys.path via conftest.py
from signal_ipc import (
    IPC_HOST,
    SignalIpcClient,
    SignalIpcError,
    SignalIpcServer,
    wait_for_server,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ephemeral_port():
    """Grab a free TCP port the OS guarantees is unused."""
    import socket as _sock
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    s.bind((IPC_HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def running_server(ephemeral_port):
    """Start a SignalIpcServer with a synthetic source_fn, stop after test.

    Source_fn returns a deterministic ramp array (i*1.0) so tests can
    assert on values.
    """
    call_log = {"count": 0}

    def source_fn(channel: str, n: int):
        call_log["count"] += 1
        if channel == "EMPTY":
            return np.zeros(0, dtype=np.float32), True, 100.0
        if channel == "BAD":
            raise RuntimeError("simulated source crash")
        # Normal: ramp 0..n-1
        arr = np.arange(n, dtype=np.float32) + call_log["count"] * 1000.0
        return arr, False, 100.0

    server = SignalIpcServer(source_fn=source_fn, port=ephemeral_port)
    server.start()
    # Wait until accept loop is ready
    assert wait_for_server(port=ephemeral_port, timeout=5.0), \
        f"server did not come up on port {ephemeral_port}"
    yield server, call_log
    server.stop()


# ---------------------------------------------------------------------------
# C18 — IPC server must bind 127.0.0.1 (never 0.0.0.0)
# ---------------------------------------------------------------------------

def test_c18_ipc_server_binds_localhost_only():
    """Contract C18: IPC must never be exposed externally."""
    # Default host constant
    assert IPC_HOST == "127.0.0.1"
    # Server uses the constant
    s = SignalIpcServer(source_fn=lambda c, n: (np.zeros(n, np.float32), False, 100.0))
    try:
        assert s.host == "127.0.0.1"
    finally:
        pass  # no start() — no resource to clean


# ---------------------------------------------------------------------------
# Round-trip: client.read → server → response
# ---------------------------------------------------------------------------

def test_basic_round_trip_returns_expected_array(running_server):
    server, _ = running_server
    port = server.port
    client = SignalIpcClient(port=port)

    raw, exhausted, sample_rate = client.read("C-1", 512)

    assert raw.dtype == np.float32
    assert len(raw) == 512
    # First call → ramp 0..511 (+ 1000 offset from call_log)
    np.testing.assert_allclose(raw[:5], [1000, 1001, 1002, 1003, 1004])
    assert exhausted is False
    assert sample_rate == 100.0


def test_second_call_advances_state(running_server):
    """server's source_fn is stateful (call_log) — each call must advance it."""
    server, call_log = running_server
    port = server.port
    client = SignalIpcClient(port=port)

    raw1, _, _ = client.read("C-1", 4)
    raw2, _, _ = client.read("C-1", 4)

    # Second call's offset must be +1000 higher (call_log incremented)
    assert raw2[0] == raw1[0] + 1000
    assert call_log["count"] == 2


def test_empty_response_carries_exhausted_flag(running_server):
    """Channel 'EMPTY' returns zero-length array + exhausted=True."""
    server, _ = running_server
    client = SignalIpcClient(port=server.port)

    raw, exhausted, _ = client.read("EMPTY", 512)

    assert len(raw) == 0
    assert exhausted is True


def test_unknown_channel_returns_error(running_server):
    """Channels not in source_fn's domain produce a structured error."""
    server, _ = running_server
    client = SignalIpcClient(port=server.port)

    # 'UNKNOWN' is not 'EMPTY'/'BAD' → source_fn returns normal ramp,
    # so this test uses a deliberately bad channel by stopping the server
    # and starting one whose source_fn raises KeyError.
    server.stop()

    def strict_source(channel, n):
        if channel not in {"C-1", "D-14"}:
            raise KeyError(channel)
        return np.zeros(n, np.float32), False, 100.0

    server2 = SignalIpcServer(source_fn=strict_source, port=server.port)
    server2.start()
    try:
        assert wait_for_server(port=server.port, timeout=5.0)
        client2 = SignalIpcClient(port=server.port)
        with pytest.raises(SignalIpcError) as exc_info:
            client2.read("NOPE", 16)
        # Error message should mention the unknown channel
        assert "unknown channel" in str(exc_info.value)
    finally:
        server2.stop()


def test_source_exception_returns_error(running_server):
    """If source_fn raises a non-KeyError Exception, IPC must not crash."""
    server, _ = running_server
    client = SignalIpcClient(port=server.port)

    with pytest.raises(SignalIpcError) as exc_info:
        client.read("BAD", 16)
    assert "source error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Protocol robustness
# ---------------------------------------------------------------------------

def test_invalid_json_request_returns_error(running_server):
    """Malformed JSON should produce an error response, not hang."""
    server, _ = running_server
    import socket as _sock
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((server.host, server.port))
    try:
        s.sendall(b"{not valid json\n")
        # Read response
        buf = b""
        while b"\nEND\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        text = buf.decode("utf-8", errors="replace")
        assert "error" in text
        assert "invalid json" in text
    finally:
        s.close()


def test_missing_fields_returns_error(running_server):
    """Request missing 'channel' or 'n' should be rejected."""
    server, _ = running_server
    import socket as _sock
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((server.host, server.port))
    try:
        # No 'channel' field
        s.sendall(b'{"n": 16}\n')
        buf = b""
        while b"\nEND\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        text = buf.decode("utf-8", errors="replace")
        assert "error" in text
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_server_stop_is_clean(ephemeral_port):
    """stop() must close the listening socket and join the thread."""
    server = SignalIpcServer(
        source_fn=lambda c, n: (np.zeros(n, np.float32), False, 100.0),
        port=ephemeral_port,
    )
    server.start()
    assert wait_for_server(port=ephemeral_port, timeout=5.0)
    server.stop()

    # Thread should be gone
    assert server._thread is None or not server._thread.is_alive()
    # Socket should be closed
    assert server._sock is None


def test_wait_for_server_returns_false_on_timeout(ephemeral_port):
    """When no server is running, wait_for_server must time out (not hang)."""
    # Use a port we know has nothing on it
    result = wait_for_server(port=ephemeral_port, timeout=0.3)
    assert result is False
