"""Space-segment CLI — on-orbit processing node.

Run independently from the ground segment::

    python -m space.main

The data acquisition card (DAQ) and sensor configuration are defined in the
DAQ_CONFIG dictionary below.  No command-line arguments — change the config
dictionary and restart to reconfigure.

Stop with Ctrl+C.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "..", ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("space")


# ===========================================================================
# DAQ card hardware configuration
# ===========================================================================
# These are HARDWARE parameters — the ground segment cannot change them.
# Each channel represents a physical input on the data acquisition card.
# Modify this dictionary and restart the space segment to reconfigure.
DAQ_CONFIG = {
    "sample_rate": 100.0,     # Hz — global acquisition rate
    "window_size": 512,        # samples per read() call
    "host": "0.0.0.0",
    "port": 9876,

    # Physical channels on the DAQ card
    "channels": [
        # Each entry: {id, source_id, loop, signal_freq_hz(optional), enabled}
        {"id": 0, "source_id": "file:NASA-MSL/C-1",   "loop": True,  "enabled": True},
        {"id": 1, "source_id": "file:NASA-MSL/D-14",  "loop": True,  "enabled": True},
        {"id": 2, "source_id": "virtual:sine",         "loop": False, "signal_freq_hz": 2.0, "enabled": True},
        {"id": 3, "source_id": "virtual:multi_sine",   "loop": False, "signal_freq_hz": 5.0, "enabled": True},
    ],
}


def main():
    # Minimal CLI: --source overrides DAQ_CONFIG (for E2E tests / quick debug)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None,
                   help="Override DAQ_CONFIG with a single source_id")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--sample-rate", type=float, default=None)
    p.add_argument("--window", type=int, default=None)
    args, _ = p.parse_known_args()

    cfg = DAQ_CONFIG.copy()
    if args.host: cfg["host"] = args.host
    if args.port: cfg["port"] = args.port
    if args.sample_rate: cfg["sample_rate"] = args.sample_rate
    if args.window: cfg["window_size"] = args.window

    window = cfg["window_size"]
    sr = cfg["sample_rate"]
    host = cfg["host"]
    port = cfg["port"]

    # If --source is given, override to single-channel mode
    if args.source:
        channels = [{"id": 0, "source_id": args.source, "loop": True, "enabled": True}]
    else:
        channels = [ch for ch in cfg["channels"] if ch.get("enabled", True)]

    if not channels:
        logger.error("No enabled channels")
        return

    from sensor_source import create_source, SensorNoiseConfig
    from preprocessing import SpacePreprocessor

    # Build sources and preprocessors for each channel
    sources: list = []
    preprocessors: list = []
    ch_ids: list = []
    for ch in channels:
        src = create_source(
            source_id=ch["source_id"],
            sample_rate=sr,
            loop=ch.get("loop", False),
            signal_freq_hz=ch.get("signal_freq_hz"),
        )
        pp = SpacePreprocessor()
        pp.fit_transform(src.read(window))  # initial scaler fit per channel
        sources.append(src)
        preprocessors.append(pp)
        ch_ids.append(ch["id"])

    # Single TSPulse detector shared across all channels
    detector = None
    try:
        from anomaly_detection import AnomalyDetector
        logger.info("Loading TSPulse detector …")
        detector = AnomalyDetector(device="cpu")
        logger.info("TSPulse loaded (%d params)", detector.n_params)
    except Exception as e:
        logger.warning("Detection disabled: %s", e)

    # Layer-1 classic filter — lightweight pre-screening before TSPulse.
    # Constant channels (std < ε) are skipped to save a full forward pass.
    from classic_filter import SpaceClassicFilter
    l1_filter = SpaceClassicFilter()
    logger.info("Layer-1 classic filter enabled (constant_std=%s)",
                l1_filter.constant_std)
    # Device tree (synced from ground, enriches telemetry with display names)
    device_tree: list = []
    tree_lock = __import__('threading').Lock()
    # Start TCP server
    from comm import SpaceServer
    server = SpaceServer(host=host, port=port)

    # Register source_id → channel mapping so the server can drain
    # per-channel buffers independently (each ground poll for one source
    # no longer clears other sources' data).
    for src, ch in zip(sources, channels):
        server.register_source(ch["source_id"], src.channel_name)

    def _on_config(cfg: dict):
        nonlocal device_tree
        if "device_tree" in cfg:
            with tree_lock:
                device_tree = cfg["device_tree"]
            logger.info("Device tree updated from ground (%d nodes)",
                        sum(1 + len(n.get("children", [])) for n in device_tree))

    server.set_on_config(_on_config)
    server.start()

    logger.info("Space node started: %d channels, sr=%.1f Hz, window=%d → tcp://%s:%d",
                len(sources), sr, window, host, port)
    for i, (src, ch) in enumerate(zip(sources, channels)):
        logger.info("  ch[%d]: %s (loop=%s)", ch["id"], src.channel_name, ch.get("loop", False))

    step = 0
    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while running:
            any_data = False
            for i, (src, pp) in enumerate(zip(sources, preprocessors)):
                raw = src.read(window)
                if len(raw) == 0:
                    continue  # loop source will never hit this
                any_data = True
                # Record the acquisition moment so ground can stamp each
                # sample with t_acq + i/sr (strict equidistant) instead of
                # back-calculating from its own wall-clock (which produces
                # fake gaps when data buffers in TCP).
                t_acq = time.time()

                cleaned = pp.transform(raw) if pp._scaler is not None else pp.fit_transform(raw)
                scores = None
                l1_decision = "pass"
                l1_detail: dict = {}

                # --- Layer 1: classic pre-filter ---------------------------
                try:
                    imputed = pp._impute(raw.astype(np.float64)).astype(np.float32)
                    l1_decision, l1_detail = l1_filter.check(imputed)
                except Exception:
                    logger.warning("L1 filter failed ch[%d]", ch_ids[i], exc_info=True)
                    l1_detail = {"error": "l1_failed"}

                # --- Layer 2: TSPulse (skipped for constant channels) -------
                if l1_decision == "skip":
                    # Constant / broken channel — TSPulse is skipped to save
                    # CPU, but the anomaly score MUST still be set (to zeros),
                    # not None.  A None score leaves a gap in the chart's
                    # anomaly-score curve and makes the channel look broken.
                    scores = np.zeros(len(raw), dtype=np.float32)
                    if step == 0:
                        logger.info("ch[%d] %s: L1 skip (%s), TSPulse skipped, scores=zeros",
                                    ch_ids[i], src.channel_name,
                                    l1_detail.get("reason", "?"))
                elif detector is not None:
                    try:
                        scores = detector.detect(imputed)
                    except Exception:
                        logger.warning("Detection failed ch[%d]", ch_ids[i], exc_info=True)

                # Look up device tree metadata for this channel
                with tree_lock:
                    _tree_meta = {}
                    _src_id = ch.get("source_id", "")
                    for _root in device_tree:
                        for _child in _root.get("children", []):
                            if _child.get("sourceId") == _src_id or \
                               _child.get("source_id") == _src_id:
                                _tree_meta = {
                                    "display_name": _child.get("name", src.channel_name),
                                    "rack": _root.get("name", ""),
                                    "description": _child.get("description", ""),
                                }
                                break
                        if _tree_meta:
                            break

                server.enqueue_telemetry(
                    channel=src.channel_name,
                    raw_values=raw,
                    scores=scores,
                    sample_rate=sr,
                    step=step,
                    exhausted=src.exhausted,
                    tree_meta=_tree_meta,
                    l1_decision=l1_decision,
                    l1_detail=l1_detail,
                    t_acq_start=t_acq,
                )

                if scores is not None and len(scores) > 0:
                    mx = float(np.nanmax(scores))
                    if mx > 0.5:
                        server.enqueue_alert(
                            channel=src.channel_name,
                            score=mx,
                            step=step,
                        )

            if not any_data:
                logger.info("All sources exhausted")
                break

            step += window

            # Pace to simulate real-time
            if sr > 0:
                interval = window / sr
                wait_until = time.time() + interval
                while running and time.time() < wait_until:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        logger.info("Space node stopped (total steps: %d)", step)


if __name__ == "__main__":
    main()
