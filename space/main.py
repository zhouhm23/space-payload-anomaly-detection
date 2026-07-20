"""Space-segment CLI — on-orbit DAQ card processing node.

Run independently from the ground segment::

    python -m space.main

架构（M1.2 双进程改造后）
========================
本进程是「采集卡」，只负责硬件拓扑 + 三层级联检测 + TCP 下发地基。
原始数据由独立的「信号发生器」进程（``signal_generator.py``）产生，
通过本地 IPC（``signal_ipc.py``，127.0.0.1:9878）按需 pull。

配置
====
硬件拓扑读自 ``space/data/space_daq.json``（通道 id/name/enabled/isSpecial
+ sample_rate + window_size + host/port）。数据源绑定（哪个通道接什么
SensorSource）由信号发生器自己的 ``signal_sources.json`` 决定，本进程
不关心。

停止：Ctrl+C。
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_HERE.parent / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Force offline mode: models are pre-cached in src/.hf_cache.  Without this
# transformers still pings huggingface.co on every startup to check for updates
# — slow (multi-second SSL handshake) and, when the network/SSL fails, it
# triggers a meta-tensor fallback that corrupts model construction.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("space")

# ===========================================================================
# Anomaly-scoring constants
# ===========================================================================
# Alert threshold — a measured-block anomaly score above this triggers an
# alert packet downstream.  Tuned at 0.5 (clip-normalised pipeline scores):
# normal sine blocks stay ~0.3-0.4 (no false alarm), multi_sine ~0.45 (low
# false alarm ~14%), while genuine anomalies like MSL T-4 reach ~0.68 and
# C-1 ~0.50.  This is a compromise — see experiments notes for the full
# sweep (0.5 gives sine 0% FP, multi_sine 14% FP, C-1 67% block recall).
ALERT_THRESHOLD: float = 0.5

# ===========================================================================
# Config path
# ===========================================================================
_DAQ_CONFIG_PATH = _HERE / "data" / "space_daq.json"


def load_daq_config(config_path: Path = _DAQ_CONFIG_PATH) -> dict:
    """Load DAQ hardware config from JSON.

    Structure must match ``space/data/space_daq.json``.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"DAQ config not found: {config_path}\n"
            f"Create one based on space_daq.json structure."
        )
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for required in ("sample_rate", "window_size", "host", "port", "channels"):
        if required not in cfg:
            raise ValueError(f"DAQ config missing required key: {required}")
    return cfg


def main():
    import argparse
    p = argparse.ArgumentParser(description="DAQ card processing node (space segment).")
    p.add_argument("--daq-config", type=str, default=str(_DAQ_CONFIG_PATH),
                   help=f"Path to space_daq.json (default: {_DAQ_CONFIG_PATH})")
    p.add_argument("--host", default=None,
                   help="Override DAQ config host (TCP listen address)")
    p.add_argument("--port", type=int, default=None,
                   help="Override DAQ config port (TCP listen port)")
    p.add_argument("--ipc-port", type=int, default=9878,
                   help="Signal-generator IPC port on 127.0.0.1 (default 9878)")
    p.add_argument("--sample-rate", type=float, default=None,
                   help="Override DAQ config sample_rate")
    p.add_argument("--window", type=int, default=None,
                   help="Override DAQ config window_size")
    p.add_argument("--channels", type=str, default=None,
                   help="Comma-separated channel names to enable (default: all enabled)")
    args, _ = p.parse_known_args()

    # ── Load DAQ config ────────────────────────────────────────────────
    cfg = load_daq_config(Path(args.daq_config))
    if args.host is not None: cfg["host"] = args.host
    if args.port is not None: cfg["port"] = args.port
    if args.sample_rate is not None: cfg["sample_rate"] = args.sample_rate
    if args.window is not None: cfg["window_size"] = args.window

    window = int(cfg["window_size"])
    sr = float(cfg["sample_rate"])
    host = cfg["host"]
    port = int(cfg["port"])

    # Filter channels by enabled flag + optional --channels whitelist
    channels = [ch for ch in cfg["channels"] if ch.get("enabled", True)]
    if args.channels:
        wanted = {c.strip() for c in args.channels.split(",") if c.strip()}
        channels = [ch for ch in channels if ch.get("name") in wanted]

    if not channels:
        logger.error("No enabled channels (check space_daq.json or --channels)")
        return

    # ── Lazy imports (keep --help fast) ────────────────────────────────
    from preprocessing import SpacePreprocessor
    from classic_filter import SpaceClassicFilter
    from comm import SpaceServer
    from signal_ipc import SignalIpcClient, wait_for_server

    # ── Connect to signal generator (IPC client) ───────────────────────
    logger.info("Connecting to signal generator on 127.0.0.1:%d ...", args.ipc_port)
    if not wait_for_server(port=args.ipc_port, timeout=15.0):
        logger.error(
            "Signal generator not reachable on 127.0.0.1:%d. "
            "Start it first: python -m space.signal_generator --port %d",
            args.ipc_port, args.ipc_port,
        )
        return
    ipc_client = SignalIpcClient(port=args.ipc_port)
    logger.info("Connected to signal generator.")

    # ── Build preprocessors (one per channel) + initial scaler fit ─────
    # The fit_transform call consumes the first window from each channel
    # via IPC.  This is intentional — the scaler must see a representative
    # block of real data before detection begins.
    preprocessors: list = []
    for ch in channels:
        ch_name = ch["name"]
        try:
            init_raw, _, _ = ipc_client.read(ch_name, window)
        except Exception as e:
            logger.error("IPC read failed for channel '%s' during scaler fit: %s", ch_name, e)
            return
        if len(init_raw) < window:
            logger.error(
                "Channel '%s' returned %d < %d samples for initial fit — "
                "source exhausted before detection could start.",
                ch_name, len(init_raw), window,
            )
            return
        pp = SpacePreprocessor()
        pp.fit_transform(init_raw)
        preprocessors.append(pp)

    # ── Load TSPulse detector (shared across channels, thread-safe) ────
    detector = None
    try:
        from anomaly_detection import AnomalyDetector
        logger.info("Loading TSPulse detector …")
        detector = AnomalyDetector(device="cpu")
        logger.info("TSPulse loaded (%d params)", detector.n_params)
    except Exception as e:
        logger.warning("Detection disabled: %s", e)

    # ── Layer-1 classic filter ─────────────────────────────────────────
    l1_filter = SpaceClassicFilter()
    logger.info("Layer-1 classic filter enabled (constant_std=%s)",
                l1_filter.constant_std)

    # ── Device tree cache (synced from ground, enriches telemetry) ─────
    device_tree: list = []
    tree_lock = threading.Lock()

    # ── Start TCP server (DAQ → ground) ────────────────────────────────
    server = SpaceServer(host=host, port=port)

    # Register channel name mapping for per-channel buffer filtering.
    # NOTE: with the IPC split, the "source_id" the ground sends is the
    # channel NAME (not the file:NASA-MSL/... string).  We map name→name
    # so SpaceServer's existing filter logic works unchanged.
    for ch in channels:
        server.register_source(ch["name"], ch["name"])

    def _on_config(cfg: dict):
        nonlocal device_tree
        if "device_tree" in cfg:
            with tree_lock:
                device_tree = cfg["device_tree"]
            logger.info("Device tree updated from ground (%d nodes)",
                        sum(1 + len(n.get("children", [])) for n in device_tree))

    server.set_on_config(_on_config)
    server.start()

    logger.info("Space DAQ node started: %d channels, sr=%.1f Hz, window=%d → tcp://%s:%d",
                len(channels), sr, window, host, port)
    for i, ch in enumerate(channels):
        logger.info("  ch[%d]: %s (special=%s)",
                    ch.get("id", i), ch["name"], ch.get("isSpecial", False))

    step = 0
    # Per-channel previous-block cache for TSPulse pipeline overlap context.
    # Keyed by channel name; value is the previous raw block (np.ndarray).
    _prev_blocks: dict[str, np.ndarray] = {}
    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Per-channel worker (runs in parallel via ThreadPoolExecutor) ────
    def _process_channel(ipc, pp, ch_dict, ch_id, cur_step):
        """IPC read → L1 → L2 → enqueue for one channel. Returns True if data was read.

        Contract C2 (修正后): ``t_acq = time.time()`` MUST be recorded
        BEFORE the IPC request, not after.  Otherwise IPC round-trip
        latency (ms-level, but variable) leaks into the acquisition
        timestamp and breaks the equidistant grid downstream.
        """
        ch_name = ch_dict["name"]

        # ★ C2 修正：先打时间戳，再发起 IPC 请求
        t_acq = time.time()
        try:
            raw, exhausted, _ = ipc.read(ch_name, window)
        except Exception as e:
            logger.warning("IPC read failed ch[%d] '%s': %s", ch_id, ch_name, e)
            return False

        if len(raw) == 0:
            return False  # source exhausted

        scores = None
        l1_decision = "pass"
        l1_detail: dict = {}

        # --- Layer 1: classic pre-filter -------------------------------
        try:
            imputed = pp._impute(raw.astype(np.float64)).astype(np.float32)
            l1_decision, l1_detail = l1_filter.check(imputed)
        except Exception:
            logger.warning("L1 filter failed ch[%d]", ch_id, exc_info=True)
            l1_detail = {"error": "l1_failed"}

        # --- Layer 2: TSPulse (skipped for constant channels) -----------
        if l1_decision == "skip":
            # ★ C3: SKIP branch must assign scores = np.zeros (not None)
            scores = np.zeros(len(raw), dtype=np.float32)
            if cur_step == 0:
                logger.info("ch[%d] %s: L1 skip (%s), TSPulse skipped, scores=zeros",
                            ch_id, ch_name, l1_detail.get("reason", "?"))
        elif detector is not None:
            try:
                # Pass previous block as context for pipeline overlap.
                ctx = _prev_blocks.get(ch_name)
                scores = detector.detect(imputed, context=ctx)
                _prev_blocks[ch_name] = imputed.copy()
            except Exception:
                logger.warning("Detection failed ch[%d]", ch_id, exc_info=True)

        # Look up device tree metadata for this channel.
        _src_id = ch_name
        with tree_lock:
            _tree_meta = {}
            for _root in device_tree:
                for _child in _root.get("children", []):
                    if _child.get("sourceId") == _src_id or \
                       _child.get("source_id") == _src_id or \
                       _child.get("name") == _src_id:
                        _tree_meta = {
                            "display_name": _child.get("name", ch_name),
                            "rack": _root.get("name", ""),
                            "description": _child.get("description", ""),
                        }
                        break
                if _tree_meta:
                    break

        # ★ C1: t_acq_start is a top-level field (handled by comm.SpaceServer)
        server.enqueue_telemetry(
            channel=ch_name,
            raw_values=raw,
            scores=scores,
            sample_rate=sr,
            step=cur_step,
            exhausted=exhausted,
            tree_meta=_tree_meta,
            l1_decision=l1_decision,
            l1_detail=l1_detail,
            t_acq_start=t_acq,
        )

        # ★ C12: AlertPacket must carry raw_window + score_window snapshots
        if scores is not None and len(scores) > 0:
            mx = float(np.nanmax(scores))
            if mx > ALERT_THRESHOLD:
                server.enqueue_alert(
                    channel=ch_name,
                    score=mx,
                    step=cur_step,
                    raw_window=raw.tolist() if hasattr(raw, 'tolist') else list(raw),
                    score_window=scores.tolist() if hasattr(scores, 'tolist') else list(scores),
                )

        return True

    from concurrent.futures import ThreadPoolExecutor

    try:
        while running:
            # Run all channels in parallel.  Phase 0 verified the shared
            # TSPulse detector is thread-safe under ThreadPoolExecutor (4
            # channels: serial 1.69s → parallel 0.36s, identical results).
            n_workers = len(channels)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(_process_channel, ipc_client, pp, ch_dict, ch_dict.get("id", i), step)
                    for i, (pp, ch_dict) in enumerate(zip(preprocessors, channels))
                ]
                results = [f.result() for f in futures]

            any_data = any(results)

            if not any_data:
                logger.info("All sources exhausted")
                break

            step += window

            # Pace to simulate real-time acquisition
            if sr > 0:
                interval = window / sr
                wait_until = time.time() + interval
                while running and time.time() < wait_until:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        logger.info("Space DAQ node stopped (total steps: %d)", step)


if __name__ == "__main__":
    main()
