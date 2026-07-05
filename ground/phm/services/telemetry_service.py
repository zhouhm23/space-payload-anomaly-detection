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
from pathlib import Path

from ..database import RingBuffer
from ..database.alert_store import AlertStore

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_GROUND_DIR = _HERE.parent.parent  # src/ground
if str(_GROUND_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUND_DIR))

from comm import GroundClient, TelemetryPacket, AlertPacket  # noqa: E402


class TelemetryService:
    """Polls the space TCP server, ingests into RingBuffer + AlertStore."""

    def __init__(
        self,
        ring: RingBuffer,
        alerts: AlertStore,
        space_host: str = "127.0.0.1",
        space_port: int = 9876,
    ) -> None:
        self.ring = ring
        self.alerts = alerts
        self.space_host = space_host
        self.space_port = space_port

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
        per-channel entry lists.  Identical to the legacy ``poll_space``."""
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

        channel_entries: dict[str, list] = {}
        alerts_list: list[dict] = []
        pkt_time = time.time()
        pkt_sr = sample_rate if sample_rate > 0 else 1.0

        for p in packets:
            if isinstance(p, TelemetryPacket):
                ch = p.channel
                raw = p.raw_values
                scores = p.scores
                n = len(raw)
                entries = channel_entries.setdefault(ch, [])
                for i in range(n):
                    sample_time = pkt_time - (n - 1 - i) / pkt_sr
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
                if p.metadata.get("exhausted", False):
                    exhausted = True
            elif isinstance(p, AlertPacket):
                alerts_list.append({
                    "channel": p.channel,
                    "score": p.score,
                    "step": p.step,
                    "message": p.message or f"异常分数 {p.score:.3f} 超阈值",
                    "time": pkt_time,
                    "type": "measured",
                })

        return channel_entries, alerts_list, exhausted


__all__ = ["TelemetryService"]
