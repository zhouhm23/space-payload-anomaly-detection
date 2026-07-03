"""Space-segment CLI — on-orbit processing node.

Run independently from the ground segment::

    cd src
    python -m space.main [--source dataset --dataset NASA-MSL --channel C-1]

Continuously reads sensor data → preprocesses → detects anomalies → buffers
results to a TCP server that the ground segment polls.

Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
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


def main():
    p = argparse.ArgumentParser(description="Space-segment processing node")
    p.add_argument("--source", choices=["dataset", "synthetic"], default="dataset")
    p.add_argument("--dataset", default="NASA-MSL")
    p.add_argument("--channel", default="C-1")
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--host", default="0.0.0.0",
                   help="TCP listen address (0.0.0.0 for cross-machine access)")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--sample-rate", type=float, default=1.0,
                   help="Sensor sample rate in Hz (-1 = bulk load all data at once)")
    p.add_argument("--no-detection", action="store_true",
                   help="Disable TSPulse anomaly detection")
    args = p.parse_args()

    # ---- initial config (overridable by ground) ----
    current_cfg = {
        "source_type": args.source,
        "dataset_name": args.dataset,
        "channel": args.channel,
        "signal_type": "multi_sine",
        "freq": 0.02,
        "noise_enabled": False,
        "missing_rate": 0.0,
        "noise_std": 0.0,
        "jitter_std": 0.0,
        "sample_rate": args.sample_rate,
        "use_detection": not args.no_detection,
    }

    from sensor_source import (
        DatasetSource, SyntheticSource, SyntheticConfig, SensorNoiseConfig,
    )
    from preprocessing import SpacePreprocessor

    def _build_source(cfg: dict):
        noise = SensorNoiseConfig(
            missing_rate=cfg.get("noise_enabled", False) and cfg.get("missing_rate", 0) or 0,
            noise_std=cfg.get("noise_enabled", False) and cfg.get("noise_std", 0) or 0,
            jitter_std=cfg.get("noise_enabled", False) and cfg.get("jitter_std", 0) or 0,
        )
        if cfg.get("source_type") == "synthetic":
            return SyntheticSource(
                config=SyntheticConfig(
                    signal_type=cfg.get("signal_type", "multi_sine"),
                    frequency=cfg.get("freq", 0.02),
                ),
                sample_rate=cfg.get("sample_rate", 1.0),
                noise=noise,
            )
        else:
            return DatasetSource(
                dataset=cfg.get("dataset_name", "NASA-MSL"),
                channel=cfg.get("channel", "C-1"),
                sample_rate=cfg.get("sample_rate", 1.0),
                noise=noise,
            )

    def _build_preproc(cfg: dict):
        return SpacePreprocessor()

    # ---- shared state (protected by lock during reconfig) ----
    import threading as _th
    _reconf_lock = _th.Lock()
    _reconf_needed = _th.Event()

    source = _build_source(current_cfg)
    preproc = _build_preproc(current_cfg)
    preproc.fit_transform(source.read(args.window))  # initial fit

    detector = None
    if current_cfg["use_detection"]:
        logger.info("Loading TSPulse detector …")
        from anomaly_detection import AnomalyDetector
        detector = AnomalyDetector(device="cpu")
        logger.info("TSPulse loaded (%d params)", detector.n_params)

    # ---- start TCP server ----
    from comm import SpaceServer
    server = SpaceServer(host=args.host, port=args.port)

    def _on_config(cfg: dict):
        nonlocal current_cfg
        with _reconf_lock:
            changed = any(current_cfg.get(k) != v for k, v in cfg.items()
                          if k in current_cfg)
            if changed:
                current_cfg.update(cfg)
                _reconf_needed.set()
                logger.info("Config updated from ground: %s",
                            {k: v for k, v in cfg.items() if k in current_cfg})

    server.set_on_config(_on_config)
    server.start()

    # ---- processing loop ----
    logger.info("Space node started [%s] window=%d → tcp://%s:%d",
                source.channel_name, args.window, args.host, args.port)
    step = 0
    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while running:
            # --- reconfigure if ground sent new config ---
            if _reconf_needed.is_set():
                with _reconf_lock:
                    source = _build_source(current_cfg)
                    preproc = _build_preproc(current_cfg)
                    preproc.fit_transform(source.read(args.window))
                    _reconf_needed.clear()
                logger.info("Reconfigured: %s ch=%s",
                            current_cfg.get("source_type"),
                            getattr(source, 'channel_name', 'N/A'))

            raw = source.read(args.window)
            if len(raw) == 0:
                # source exhausted — wait for Ctrl+C
                logger.info("Source exhausted at step %d", step)
                while running:
                    time.sleep(0.5)
                break

            step += len(raw)

            cleaned = preproc.transform(raw) if preproc._scaler is not None \
                else preproc.fit_transform(raw)

            scores = None
            if detector is not None:
                try:
                    imputed = preproc._impute(raw.astype(np.float64)).astype(np.float32)
                    scores = detector.detect(imputed)
                except Exception:
                    logger.warning("Detection failed", exc_info=True)

            server.enqueue_telemetry(
                channel=source.channel_name,
                raw_values=raw,
                scores=scores,
                step=step,
                exhausted=source.exhausted,
            )

            if scores is not None and len(scores) > 0:
                mx = float(np.nanmax(scores))
                if mx > 0.5:
                    server.enqueue_alert(
                        channel=source.channel_name,
                        score=mx,
                        step=step,
                    )

            # pace to simulate real-time (skip if sample_rate < 0)
            if source.sample_rate > 0:
                interval = args.window / source.sample_rate
                wait_until = time.time() + interval
                while running and time.time() < wait_until:
                    # wake up immediately if ground sent a new config
                    if _reconf_needed.is_set():
                        break
                    time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        logger.info("Space node stopped (total steps: %d)", step)


if __name__ == "__main__":
    main()
