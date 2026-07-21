"""Bridge between Django and the phm service container.

职责：
1. 在 Django 启动时初始化 phm.api.deps.Container（含所有 service + 模型预加载）
2. 管理 3 个后台线程：
   - auto-poll：2s 周期，从天基 TCP 拉取数据
   - eval：1s 周期，并行评估所有通道（预测+检测+预警）
   - auto-diagnosis：按需启动，LLM 批量诊断（DiagnosisService 内部线程）

设计要点（v1.1）：
- Container 初始化耗时数秒（加载 TSPulse + TTM-R3 + RUL 模型）。
  ready() 不应阻塞 Django 启动，因此初始化放在后台线程。
- 三态机：'idle' → 'initializing' → 'ready'/'failed'。
  API 端点根据状态返回 200/503，避免误导前端。
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

# 状态机：'idle' → 'initializing' → 'ready' / 'failed'
_state = 'idle'
_state_lock = threading.Lock()
_init_error: Exception | None = None

_auto_poll_stop = threading.Event()
_auto_poll_thread: threading.Thread | None = None
_eval_stop = threading.Event()
_eval_thread: threading.Thread | None = None
_init_thread: threading.Thread | None = None

_AUTO_POLL_INTERVAL = 2.0  # 秒
_AUTO_POLL_BLOCK = 512     # 默认块大小
_MAX_EVAL_WORKERS = 8      # 模型评估并行 worker 上限

# ── 天地链路 RTT 统计 ────────────────────────────────────────────────────────
# poll_one 内部测量 TCP 往返时延（发请求到收到响应），多传感器取最小值。
# 这是真实的"天地信号传输延迟"（本地测试应接近 0）。
_link_rtt_ms: float | None = None           # 最近一次 poll 的最小 RTT
_link_last_success_ts: float = 0.0           # 最近一次 poll 成功时刻
_link_fail_count: int = 0                    # 连续失败计数（连续 3 次 → 中断）
_LINK_FAIL_THRESHOLD = 3                     # 链路中断判定阈值
_link_rtt_lock = threading.Lock()


def get_link_status() -> dict:
    """返回天地链路状态（顶栏用）。

    - rtt_ms: 最近一次成功 poll 的最小 RTT（毫秒）
    - status: 'online'（RTT<3000ms 且连续失败<3）/ 'degraded' / 'offline'
    - last_success_ts: 最近成功时刻
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
    """记录一次 poll 的结果，更新链路统计。"""
    global _link_rtt_ms, _link_fail_count, _link_last_success_ts
    with _link_rtt_lock:
        if success and rtt_ms is not None:
            # 多传感器并行 poll，取最小 RTT（最快的那条路径）
            if _link_rtt_ms is None or rtt_ms < _link_rtt_ms:
                _link_rtt_ms = rtt_ms
            _link_fail_count = 0
            _link_last_success_ts = time.time()
        else:
            _link_fail_count += 1


def get_state() -> str:
    """返回当前初始化状态：'idle' / 'initializing' / 'ready' / 'failed'."""
    with _state_lock:
        return _state


def get_init_error() -> str | None:
    """初始化失败时返回错误信息字符串，否则 None。"""
    with _state_lock:
        if _init_error is None:
            return None
        return f"{type(_init_error).__name__}: {_init_error}"


def start() -> None:
    """启动 Container 初始化（后台线程，不阻塞）。幂等。"""
    global _state, _init_thread
    with _state_lock:
        if _state in ('initializing', 'ready'):
            return
        _state = 'initializing'

    # 初始化放后台线程（耗时数秒，含模型加载）
    if _init_thread is None or not _init_thread.is_alive():
        _init_thread = threading.Thread(target=_init_worker, daemon=True, name='phm-init')
        _init_thread.start()


def _init_worker() -> None:
    """后台初始化 Container + 启动 auto-poll/eval 线程。"""
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
    """返回 Container 单例。未就绪时抛 RuntimeError（调用方应先 get_state() 检查）。

    Lazy-init 兜底：即便 ready() 没在服务进程跑（如 migrate），首次请求也会触发初始化。
    """
    state = get_state()
    if state == 'failed':
        raise RuntimeError(f"PHM container init failed: {get_init_error()}")
    if state != 'ready':
        # 未就绪 → 触发初始化（如未启动），并告知调用方等待
        if state == 'idle':
            start()
        raise RuntimeError(f"PHM container not ready (state={state})")

    from phm.api import deps
    return deps.get()


def stop() -> None:
    """停止后台线程 + flush SQLite。"""
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


# ── Auto-poll 线程 ──────────────────────────────────────────────────────────
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
                # 收集本轮所有 poll 结果，更新链路状态
                if results:
                    min_rtt = min((r[0] for r in results if r[0] is not None), default=None)
                    any_success = any(r[1] for r in results)
                    _record_poll_result(min_rtt, any_success)
        except Exception:
            logger.debug("Auto-poll cycle failed", exc_info=True)
        _auto_poll_stop.wait(_AUTO_POLL_INTERVAL)


def _poll_one(c, src: str) -> tuple[float | None, bool]:
    """单传感器 poll，返回 (rtt_ms, success)。"""
    try:
        t0 = time.time()
        c.telemetry.poll(src, 100.0, _AUTO_POLL_BLOCK)
        rtt_ms = (time.time() - t0) * 1000
        return rtt_ms, True
    except Exception:
        logger.debug("Poll failed for source %s", src, exc_info=True)
        return None, False


# ── Model-eval 线程 ─────────────────────────────────────────────────────────
def _eval_loop() -> None:
    """后台循环：每周期并行评估所有通道（预测+检测+预警）。

    两阶段：
    - Phase A：per-channel eval（forecast → cascade → warning state machine）
    - Phase B：per-folder co-anomaly consensus（联合告警）

    历史教训（Day17-续）：串行 4 通道耗时 2.87s > poll 间隔 2s 导致积压。
    改并行 + torch.no_grad 后 <0.5s。
    """
    from phm.api import deps

    while not _eval_stop.is_set():
        try:
            c = deps.get()
            channels = c.ring.channels()
            if channels:
                # Phase A: 并行 per-channel eval
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

                # Phase B: per-folder co-anomaly consensus（联合告警）
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
