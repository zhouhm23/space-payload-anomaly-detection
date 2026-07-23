"""Bridge between Django and the phm service container.

Responsibilities:
1. Initialise phm.api.deps.Container at Django startup (all services + model
   preloading).
2. Manage 3 background threads:
   - auto-poll: 2s cycle, pulls data from the space-segment TCP.
   - eval: 1s cycle, evaluates all channels in parallel (forecast + detection
     + warning).
   - auto-diagnosis: started on demand, batch LLM diagnosis (internal thread
     of DiagnosisService).

Design notes (v1.1):
- Container init takes several seconds (loads TSPulse + TTM-R3 + RUL models).
  ready() must not block Django startup, so init runs in a background thread.
- Three-state machine: 'idle' → 'initializing' → 'ready'/'failed'.
  API endpoints return 200/503 by state so the front-end is not misled.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

# State machine: 'idle' → 'initializing' → 'ready' / 'failed'
_state = 'idle'
_state_lock = threading.Lock()
_init_error: Exception | None = None

_auto_poll_stop = threading.Event()
_auto_poll_thread: threading.Thread | None = None
_eval_stop = threading.Event()
_eval_thread: threading.Thread | None = None
_init_thread: threading.Thread | None = None

_AUTO_POLL_INTERVAL = 2.0  # seconds
_AUTO_POLL_BLOCK = 512     # default block size
_MAX_EVAL_WORKERS = 8      # cap on parallel model-eval workers

# ── Space-ground link RTT statistics ────────────────────────────────────────
# poll_one measures the TCP round-trip (request sent → response received) and,
# across multiple sensors, keeps the minimum. This is the real "space-ground
# signal transmission delay" (should be near 0 in local tests).
_link_rtt_ms: float | None = None           # min RTT of the most recent poll
_link_last_success_ts: float = 0.0           # timestamp of the most recent successful poll
_link_fail_count: int = 0                    # consecutive failures (3 in a row → offline)
_LINK_FAIL_THRESHOLD = 3                     # link-down threshold
_link_rtt_lock = threading.Lock()


def get_link_status() -> dict:
    """Return the space-ground link status (for the top bar).

    - rtt_ms: min RTT of the most recent successful poll (milliseconds)
    - status: 'online' (RTT<3000ms and consecutive failures<3) / 'degraded' / 'offline'
    - last_success_ts: most recent success timestamp
    """
    with _link_rtt_lock:
        rtt = _link_rtt_ms
        fails = _link_fail_count
        last_ts = _link_last_success_ts
    if fails >= _LINK_FAIL_THRESHOLD:
        return {'rtt_ms': None, 'status': 'offline', 'last_success_ts': last_ts}
    if rtt is None:
        return {'rtt_ms': None, 'status': 'waiting', 'last_success_ts': last_ts}
    if rtt < 3000:
        return {'rtt_ms': rtt, 'status': 'online', 'last_success_ts': last_ts}
    return {'rtt_ms': rtt, 'status': 'degraded', 'last_success_ts': last_ts}


def _record_poll_result(rtt_ms: float | None, success: bool) -> None:
    """Record one poll's result and update the link statistics."""
    global _link_rtt_ms, _link_fail_count, _link_last_success_ts
    with _link_rtt_lock:
        if success and rtt_ms is not None:
            # Sensors are polled in parallel; keep the min RTT (the fastest path)
            if _link_rtt_ms is None or rtt_ms < _link_rtt_ms:
                _link_rtt_ms = rtt_ms
            _link_fail_count = 0
            _link_last_success_ts = time.time()
        else:
            _link_fail_count += 1


def get_state() -> str:
    """Return the current init state: 'idle' / 'initializing' / 'ready' / 'failed'."""
    with _state_lock:
        return _state


def get_init_error() -> str | None:
    """Return the init error string on failure, else None."""
    with _state_lock:
        if _init_error is None:
            return None
        return f"{type(_init_error).__name__}: {_init_error}"


def start() -> None:
    """Start Container init (background thread, non-blocking). Idempotent."""
    global _state, _init_thread
    with _state_lock:
        if _state in ('initializing', 'ready'):
            return
        _state = 'initializing'

    # Init runs in a background thread (takes seconds, includes model loading)
    if _init_thread is None or not _init_thread.is_alive():
        _init_thread = threading.Thread(target=_init_worker, daemon=True, name='phm-init')
        _init_thread.start()


def _init_worker() -> None:
    """Background: initialise Container + start the auto-poll/eval threads."""
    global _state, _init_error
    try:
        from phm.api import deps
        deps.init(
            space_host=getattr(settings, 'SPACE_HOST', '127.0.0.1'),
            space_port=getattr(settings, 'SPACE_PORT', 9876),
            config_path=Path(getattr(settings, 'PHM_CONFIG_PATH', '')),
            device="cpu",
        )
        _start_background_threads()
        with _state_lock:
            _state = 'ready'
            _init_error = None
        logger.info("PHM services_bridge ready (auto-poll + eval threads started)")
    except Exception as e:
        with _state_lock:
            _state = 'failed'
            _init_error = e
        logger.exception("PHM services_bridge init FAILED")


def get_container():
    """Return the Container singleton. Raises RuntimeError when not ready (callers should check get_state() first).

    Lazy-init fallback: even if start() was never run in the serving process
    (e.g. during migrate), the first request triggers init.
    """
    state = get_state()
    if state == 'failed':
        raise RuntimeError(f"PHM container init failed: {get_init_error()}")
    if state != 'ready':
        # Not ready → trigger init (if not started) and tell the caller to wait
        if state == 'idle':
            start()
        raise RuntimeError(f"PHM container not ready (state={state})")

    from phm.api import deps
    return deps.get()


def stop() -> None:
    """Stop the background threads + flush SQLite."""
    global _init_thread, _auto_poll_thread, _eval_thread
    _stop_background_threads()
    if _init_thread is not None and _init_thread.is_alive():
        _init_thread.join(timeout=5.0)
    _init_thread = None
    try:
        from phm.api import deps
        deps.shutdown()
    except Exception:
        pass
    with _state_lock:
        global _state
        _state = 'idle'
    logger.info("PHM services_bridge stopped")


# ── Auto-poll thread ────────────────────────────────────────────────────────
def _auto_poll_loop() -> None:
    from phm.api import deps
    from phm.services.tree_utils import get_flat_sensors

    while not _auto_poll_stop.is_set():
        try:
            c = deps.get()
            config_data = c.config.load()
            tree = config_data.get("device_tree", [])
            sources = [s.get("sourceId") for s in get_flat_sensors(tree) if s.get("sourceId")]
            if not sources:
                _auto_poll_stop.wait(_AUTO_POLL_INTERVAL)
                continue
            with ThreadPoolExecutor(max_workers=len(sources)) as pool:
                results = list(pool.map(lambda src: _poll_one(c, src), sources))
                # Collect this round's poll results and update the link status
                if results:
                    min_rtt = min((r[0] for r in results if r[0] is not None), default=None)
                    any_success = any(r[1] for r in results)
                    _record_poll_result(min_rtt, any_success)
        except Exception:
            logger.debug("Auto-poll cycle failed", exc_info=True)
        _auto_poll_stop.wait(_AUTO_POLL_INTERVAL)


def _poll_one(c, src: str) -> tuple[float | None, bool]:
    """Poll a single sensor; return (rtt_ms, success)."""
    try:
        t0 = time.time()
        c.telemetry.poll(src, 100.0, _AUTO_POLL_BLOCK)
        rtt_ms = (time.time() - t0) * 1000
        return rtt_ms, True
    except Exception:
        logger.debug("Poll failed for source %s", src, exc_info=True)
        return None, False


# ── Model-eval thread ───────────────────────────────────────────────────────
def _eval_loop() -> None:
    """Background loop: each cycle evaluates all channels in parallel (forecast + detection + warning).

    Two phases:
    - Phase A: per-channel eval (forecast → cascade → warning state machine)
    - Phase B: per-folder co-anomaly consensus (joint alerts)

    Historical lesson (Day17 follow-up): serial 4-channel eval took 2.87s > the 2s poll
    interval, causing backlog. Going parallel + torch.no_grad brought it under 0.5s.
    """
    from phm.api import deps

    while not _eval_stop.is_set():
        try:
            c = deps.get()
            channels = c.ring.channels()
            if channels:
                # Phase A: parallel per-channel eval
                n_workers = min(len(channels), _MAX_EVAL_WORKERS)
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(c.warning_service.evaluate_channel, ch, _AUTO_POLL_BLOCK): ch
                        for ch in channels
                    }
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception:
                            logger.debug("Eval failed for %s", futures[fut], exc_info=True)

                # Phase B: per-folder co-anomaly consensus (joint alerts)
                try:
                    tree = c.config.load().get("device_tree", [])
                    joint_alerts = c.warning_service.evaluate_all_folders(tree)
                    for ja in joint_alerts:
                        c.warning_service._emit_joint_alert(ja)
                except Exception:
                    logger.debug("Joint detection failed", exc_info=True)
        except Exception:
            logger.debug("Eval cycle failed", exc_info=True)
        _eval_stop.wait(1.0)


def _start_background_threads() -> None:
    _auto_poll_stop.clear()
    _eval_stop.clear()
    global _auto_poll_thread, _eval_thread
    if _auto_poll_thread is None or not _auto_poll_thread.is_alive():
        _auto_poll_thread = threading.Thread(target=_auto_poll_loop, daemon=True, name="auto-poll")
        _auto_poll_thread.start()
    if _eval_thread is None or not _eval_thread.is_alive():
        _eval_thread = threading.Thread(target=_eval_loop, daemon=True, name="model-eval")
        _eval_thread.start()


def _stop_background_threads() -> None:
    _auto_poll_stop.set()
    _eval_stop.set()
    global _auto_poll_thread, _eval_thread
    if _auto_poll_thread is not None and _auto_poll_thread.is_alive():
        _auto_poll_thread.join(timeout=5.0)
    _auto_poll_thread = None
    if _eval_thread is not None and _eval_thread.is_alive():
        _eval_thread.join(timeout=10.0)
    _eval_thread = None
