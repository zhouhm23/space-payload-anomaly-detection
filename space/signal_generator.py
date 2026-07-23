"""Signal generator — independent process that owns SensorSource instances.

Architectural context
=====================
M1.2 splits the SensorSource calls that used to live inside ``main.py`` out
into a standalone process:

- This process (the signal generator): holds the SensorSource instances
  (FileSource / VirtualSensorSource) and answers the DAQ card's read requests
  via local TCP IPC (``signal_ipc.SignalIpcServer``).
- The DAQ card (``main.py``): requests data from this process via
  ``signal_ipc.SignalIpcClient``.

Why a separate process
======================
1. In the real world the "signal source" is a physical sensor (separate
   hardware), not a part of the DAQ card.
2. During simulation FileSource loads large datasets into memory; a separate
   process keeps it from contending with the DAQ card's TSPulse model for RAM.
3. Process isolation means a data-source switch (e.g. NASA-MSL → a real sensor
   feed) only touches this process; the DAQ card is unchanged.

Configuration
=============
Reads ``space/data/signal_sources.json`` with this shape::

    {
      "default_sample_rate": 100.0,
      "bindings": [
        {"channel": "C-1", "sourceId": "file:NASA-MSL/C-1", "loop": true},
        {"channel": "VS-sine", "sourceId": "virtual:sine", "signal_freq_hz": 2.0}
      ]
    }

- ``channel`` must match ``channels[].name`` in ``space_daq.json``.
- ``sourceId / loop / signal_freq_hz`` are all SensorSource constructor params.

Startup
=======
Run as a module::

    python -m space.signal_generator [--port 9878] [--config path]

or directly ``python space/signal_generator.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal as _sig
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ── sys.path setup (so `from sensor_source import ...` works) ──────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [signal-gen] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── default config path ────────────────────────────────────────────────
_DEFAULT_CONFIG = _HERE / "data" / "signal_sources.json"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_bindings(config_path: Path) -> tuple[float, list[dict]]:
    """Load signal-source bindings from JSON.

    Returns ``(default_sample_rate, bindings_list)``.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Signal sources config not found: {config_path}\n"
            f"Create one based on signal_sources.json structure."
        )
    text = config_path.read_text(encoding="utf-8")
    cfg = json.loads(text)
    default_sr = float(cfg.get("default_sample_rate", 100.0))
    bindings = cfg.get("bindings", [])
    if not bindings:
        raise ValueError(f"No bindings found in {config_path}")
    return default_sr, bindings


def build_sources(bindings: list[dict], default_sample_rate: float) -> dict[str, tuple]:
    """Instantiate a SensorSource for each binding.

    Returns ``{channel_name: (SensorSource, sample_rate)}``.

    The sample_rate returned alongside each source is the per-channel
    rate — bindings may override the default. Currently all bindings use
    the default (real DAQ cards share a single clock), but the field is
    kept for future per-channel rate support.
    """
    # Lazy import — keeps --help fast and avoids pulling HF/torch if user
    # is just inspecting config.
    from sensor_source import create_source  # noqa: WPS433

    sources: dict[str, tuple] = {}
    for b in bindings:
        channel = b.get("channel")
        source_id = b.get("sourceId") or b.get("source_id")
        if not channel or not source_id:
            raise ValueError(f"Binding missing 'channel' or 'sourceId': {b!r}")
        # Per-binding sample_rate override (default: global default_sample_rate)
        sr = float(b.get("sample_rate", default_sample_rate))
        loop = bool(b.get("loop", False))
        signal_freq_hz = b.get("signal_freq_hz")

        kwargs = {
            "sample_rate": sr,
            "loop": loop,
        }
        if signal_freq_hz is not None:
            kwargs["signal_freq_hz"] = float(signal_freq_hz)

        src = create_source(source_id=source_id, **kwargs)
        sources[channel] = (src, sr)
        logger.info(
            "  bound channel '%s' → %s (sr=%.1f Hz, loop=%s%s)",
            channel, source_id, sr, loop,
            f", freq={signal_freq_hz}Hz" if signal_freq_hz else "",
        )
    return sources


# ---------------------------------------------------------------------------
# Source function (passed to SignalIpcServer)
# ---------------------------------------------------------------------------

def make_source_fn(sources: dict[str, tuple]):
    """Build the (channel, n) -> (raw, exhausted, sample_rate) callback.

    The closure captures ``sources`` and protects it with a per-channel
    lock. Although the IPC server is single-threaded, the lock is cheap
    insurance against future threading changes.
    """
    locks: dict[str, threading.Lock] = {ch: threading.Lock() for ch in sources}

    def source_fn(channel: str, n: int):
        if channel not in sources:
            raise KeyError(channel)
        src, sr = sources[channel]
        lock = locks[channel]
        with lock:
            raw = src.read(n)
            exhausted = bool(src.exhausted)
        # Always return float32 — contract with IPC encoder + ground side
        if raw.dtype != np.float32:
            raw = raw.astype(np.float32)
        return raw, exhausted, sr

    return source_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal generator (owns SensorSources, serves IPC).",
    )
    parser.add_argument(
        "--port", type=int, default=9878,
        help="IPC port (default 9878, bound to 127.0.0.1 only)",
    )
    parser.add_argument(
        "--config", type=str, default=str(_DEFAULT_CONFIG),
        help=f"Path to signal_sources.json (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Validate config and exit (do not start server)",
    )
    args, _ = parser.parse_known_args()

    config_path = Path(args.config)
    logger.info("Loading signal sources from %s", config_path)
    default_sr, bindings = load_bindings(config_path)
    logger.info("Config OK: %d bindings, default sample_rate=%.1f Hz",
                len(bindings), default_sr)

    if args.check:
        # Validate bindings actually instantiate
        try:
            sources = build_sources(bindings, default_sr)
            logger.info("All %d bindings instantiated successfully.", len(sources))
        except Exception as e:
            logger.error("Binding instantiation failed: %s", e)
            sys.exit(1)
        return

    # Build sources (this loads datasets — may take a few seconds)
    logger.info("Building SensorSource instances...")
    sources = build_sources(bindings, default_sr)
    logger.info("Built %d sources.", len(sources))

    source_fn = make_source_fn(sources)

    # Lazy import of IPC server (only needed when actually serving)
    from signal_ipc import SignalIpcServer, wait_for_server  # noqa: WPS433

    server = SignalIpcServer(source_fn=source_fn, port=args.port)

    # ── graceful shutdown on SIGINT / SIGTERM ───────────────────────────
    running = {"flag": True}

    def _shutdown(sig, frame):
        logger.info("Received signal %d, shutting down...", sig)
        running["flag"] = False
        server.stop()

    _sig.signal(_sig.SIGINT, _shutdown)
    if hasattr(_sig, "SIGTERM"):
        _sig.signal(_sig.SIGTERM, _shutdown)

    server.start()
    logger.info("Signal generator ready on 127.0.0.1:%d", args.port)
    logger.info("Waiting for DAQ card to connect...")

    # Block until shutdown signal
    try:
        while running["flag"]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown(_sig.SIGINT, None)

    logger.info("Signal generator exited.")


if __name__ == "__main__":
    main()
