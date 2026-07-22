"""Telemetry ingest service.

Wraps the TCP poll (``GroundClient``) + RingBuffer ingest, preserving the
exact behaviour of the legacy ``server.py::poll_space`` + ``api_poll``
pair.  Pulling it out of the route makes it testable and lets the warning
service subscribe to ingested blocks.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import threading
from pathlib import Path

from ..database import RingBuffer, SQLiteStore
from ..database.alert_store import AlertStore

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_GROUND_DIR = _HERE.parent.parent  # src/ground
if str(_GROUND_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUND_DIR))

from comm import GroundClient, TelemetryPacket, AlertPacket  # noqa: E402


class TelemetryService:
    """Polls the space TCP server, ingests into RingBuffer + AlertStore + SQLite."""

    def __init__(
        self,
        ring: RingBuffer,
        alerts: AlertStore,
        sqlite: SQLiteStore | None = None,
        space_host: str = "127.0.0.1",
        space_port: int = 9876,
    ) -> None:
        self.ring = ring
        self.alerts = alerts
        self.sqlite = sqlite
        self.space_host = space_host
        self.space_port = space_port
        # Track the last assigned timestamp per channel so consecutive
        # polls produce non-overlapping time ranges.  Without this, each
        # poll back-calculates from time.time(), and since the poll
        # interval (6s) is close to the block duration (5.12s), the
        # back-calculated ranges overlap heavily, causing two data
        # streams to interleave in SQLite (visual zig-zag in the chart).
        self._last_ts: dict[str, float] = {}
        # Guards _last_ts read-modify-write during parallel auto-poll.
        # Each source maps to a distinct channel, but the dict access is a
        # compound operation (read → compute → write) so we serialise it.
        self._ts_lock = threading.Lock()

    def poll(
        self,
        source_id: str = "file:NASA-MSL/C-1",
        sample_rate: float = 50.0,
        block_size: int = 512,
    ) -> dict:
        """One poll cycle.  Returns the same dict shape as the legacy
        ``api_poll`` response so routes can forward it verbatim."""
        channel_entries, alerts_list, exhausted = self._poll_space(
            source_id, sample_rate
        )

        # Persist into ring buffer
        self.ring.ingest(channel_entries)
        self.alerts.extend(alerts_list)

        # Persist into SQLite (async batch — non-blocking)
        if self.sqlite is not None:
            for ch, entries in channel_entries.items():
                self.sqlite.enqueue_telemetry_batch(entries)
            for alert in alerts_list:
                self.sqlite.enqueue_alert(alert)

        # Slice latest block per channel for the response
        channels = self.ring.snapshot_block(block_size)

        return {
            "channels": channels,
            "alerts": alerts_list,
            "exhausted": exhausted,
            "total": self.ring.total_points(),
            "block_size": block_size,
            # Raw entries per channel (for the warning/health services)
            "_ingested": channel_entries,
        }

    # -- internal: ported verbatim from legacy server.py -------------------

    def _poll_space(
        self,
        source_id: str = "file:NASA-MSL/C-1",
        sample_rate: float = 50.0,
    ) -> tuple[dict[str, list], list[dict], bool]:
        """Connect to space TCP, drain buffered packets, reshape into
        per-channel entry lists.  Identical to the legacy ``poll_space``.

        Raises ``ConnectionError`` when the space TCP server is unreachable
        (port not listening / SYN timeout).  This lets callers distinguish
        "连接失败" from "连接成功但本周期无数据"——前者应被 ``_poll_one``
        记为链路失败，后者是合法空 poll（不影响链路状态）。

        历史教训（Day20）：``GroundClient.poll()`` 内部吞掉所有 socket
        异常返回空 list，导致 ``_poll_one`` 把连接超时（~2s）误判为
        "链路 RTT=2000ms 且 success=True"，``link_status`` 恒显示 online。
        修复：``_poll_space`` 调 ``GroundClient.poll()`` 后检查
        ``client.connected`` 标志（``GroundClient.poll()`` 内部维护，
        True=连上，False=socket 异常），False 时抛 ``ConnectionError``。
        """
        exhausted = False
        try:
            client = GroundClient(host=self.space_host, port=self.space_port, timeout=2)
            packets = client.poll({
                "source_id": source_id,
                "sample_rate": sample_rate,
                "use_detection": True,
            })
        except Exception:
            return {}, [], False

        # 连接失败冒泡：GroundClient.poll() 吞了 socket 异常返回空 list，
        # 但 client.connected 标志会暴露真实状态。连不上时抛 ConnectionError
        # 让 _poll_one 走 except → success=False（修复 Day20 link_status bug）。
        # 兼容老版 GroundClient（无 connected 属性）：默认 True 不影响。
        if getattr(client, 'connected', True) is False:
            raise ConnectionError(
                f"space TCP {self.space_host}:{self.space_port} unreachable"
            )

        channel_entries: dict[str, list] = {}
        alerts_list: list[dict] = []
        pkt_sr = sample_rate if sample_rate > 0 else 1.0

        # When a poll returns MULTIPLE packets for the same channel (which
        # happens when the space segment has buffered several blocks since
        # the last poll), each packet must get a SEPARATE, non-overlapping
        # timestamp range.  Previously each packet used ``time.time()``
        # independently — but multiple packets arrive within the same
        # millisecond, so their back-calculated timestamps overlapped,
        # producing interleaved/duplicate data in SQLite (visual zig-zag).
        #
        # CRITICAL: We also must not overlap with the PREVIOUS poll's
        # timestamps.  The auto-poll interval (6s) is close to the block
        # duration (5.12s @ 100Hz × 512), so back-calculating from
        # time.time() every poll produces heavily overlapping ranges
        # across consecutive polls.  We track ``self._last_ts`` per channel
        # and ensure each new poll's timestamps start AFTER the last one.
        total_samples_by_channel: dict[str, int] = {}
        for p in packets:
            if isinstance(p, TelemetryPacket):
                total_samples_by_channel[p.channel] = (
                    total_samples_by_channel.get(p.channel, 0) + len(p.raw_values)
                )

        now = time.time()

        # Track how many samples we've already assigned for each channel
        assigned_by_channel: dict[str, int] = {}

        for p in packets:
            if isinstance(p, TelemetryPacket):
                ch = p.channel
                raw = p.raw_values
                scores = p.scores
                n = len(raw)
                entries = channel_entries.setdefault(ch, [])

                total_for_ch = total_samples_by_channel[ch]
                already = assigned_by_channel.get(ch, 0)

                # Timestamp strategy:
                # - If space sent t_acq_start (newer builds), anchor the
                #   channel timeline ONCE at the first packet's collection
                #   moment; every subsequent sample (within this packet,
                #   across packets in the same poll, and across polls)
                #   advances strictly by 1/pkt_sr off the previous sample.
                #   This yields a perfectly equidistant grid with no fake
                #   gaps, regardless of TCP buffering jitter or pacing
                #   imprecision — exactly what pred needs for row alignment.
                # - Otherwise (old space builds), fall back to the legacy
                #   wall-clock back-calculation with _last_ts anti-overlap.
                with self._ts_lock:
                    if p.t_acq_start is not None:
                        prev_last = self._last_ts.get(ch)
                        # base = previous sample + 1 step (seamless), or the
                        # true collection time if this is the very first
                        # packet for the channel.
                        base = (prev_last + 1.0 / pkt_sr) if prev_last is not None \
                            else p.t_acq_start
                        for i in range(n):
                            sample_time = base + i / pkt_sr
                            entries.append({
                                "raw": float(raw[i]),
                                "score": (
                                    float(scores[i])
                                    if scores is not None and i < len(scores)
                                    else None
                                ),
                                "received_at": sample_time,
                                "channel": ch,
                            })
                    else:
                        # Legacy: back-calculate from wall-clock now.
                        batch_span = (total_for_ch - 1) / pkt_sr
                        prev_last = self._last_ts.get(ch)
                        if prev_last is not None and now - batch_span <= prev_last:
                            # Overlap detected — shift the right edge to just
                            # after the previous batch's last timestamp.
                            ref_time = prev_last + total_for_ch / pkt_sr
                        else:
                            ref_time = now

                        for i in range(n):
                            global_idx = already + i
                            sample_time = ref_time - (total_for_ch - 1 - global_idx) / pkt_sr
                            entries.append({
                                "raw": float(raw[i]),
                                "score": (
                                    float(scores[i])
                                    if scores is not None and i < len(scores)
                                    else None
                                ),
                                "received_at": sample_time,
                                "channel": ch,
                            })

                    assigned_by_channel[ch] = already + n

                    # Record the last timestamp for this channel so the next
                    # poll can avoid overlapping.
                    if entries:
                        self._last_ts[ch] = entries[-1]["received_at"]

                if p.metadata.get("exhausted", False):
                    exhausted = True
            elif isinstance(p, AlertPacket):
                # 优先用 space 段传来的真实异常采样时刻（acq_ts），
                # 它与遥测时间轴同源（t_acq_start 锚定），前端红点能精准对齐。
                # 旧版 space 段不传 acq_ts，兜底用接收时刻 time.time()。
                alert_time = p.acq_ts if p.acq_ts is not None else time.time()
                alerts_list.append({
                    "channel": p.channel,
                    "score": p.score,
                    "step": p.step,
                    "message": p.message or f"异常分数 {p.score:.3f} 超阈值",
                    "time": alert_time,
                    "type": "measured",
                    "raw_snapshot": p.raw_window,
                    "score_snapshot": p.score_window,
                })

        return channel_entries, alerts_list, exhausted


__all__ = ["TelemetryService"]
