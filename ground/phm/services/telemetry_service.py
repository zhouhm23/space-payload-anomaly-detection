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
        "connection failed" from "connected but no data this cycle" — the
        former should be recorded as a link failure by ``_poll_one``, while
        the latter is a valid empty poll (does not affect link status).

        Historical lesson (Day20): ``GroundClient.poll()`` silently swallowed
        all socket exceptions and returned an empty list, causing ``_poll_one``
        to misinterpret a connection timeout (~2s) as "link RTT=2000ms and
        success=True", keeping ``link_status`` permanently online.
        Fix: after calling ``GroundClient.poll()``, check the ``client.connected``
        flag (maintained internally by ``GroundClient.poll()``, True = connected,
        False = socket exception) and raise ``ConnectionError`` when False.

        Compatible with older GroundClient versions (no ``connected`` attribute):
        defaults to True, so behaviour is unchanged.
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

        # Bubble up connection failures: GroundClient.poll() swallows socket exceptions
        # and returns an empty list, but the client.connected flag reveals the true
        # state.  When unreachable, raise ConnectionError so _poll_one takes the
        # except path → success=False (fixes Day20 link_status bug).
        # Compatible with legacy GroundClient (no connected attribute): defaults to
        # True so behaviour is unchanged.
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
                # Prefer the real anomaly-sample timestamp (acq_ts) sent by the
                # space segment — it shares the same timeline anchor (t_acq_start)
                # as telemetry, so the frontend red dots align precisely.
                # Legacy space builds do not send acq_ts; fall back to the
                # reception time time.time().
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
