"""Real-time RingBuffer for streaming telemetry.

Replaces the flat ``ring_buffers: dict[str, list]`` + ``buffer_lock`` pair
that lived in ``server.py``.  The data shape and slice semantics are kept
byte-for-byte identical so ``/api/poll`` consumers see no behaviour change.

Each entry: ``{"raw": float, "score": float | None, "received_at": float,
"channel": str}`` — exactly the shape produced by the legacy poll loop.
"""

from __future__ import annotations

import threading
import time

from ..config import RING_BUFFER_MAX


class ChannelStore:
    """A single channel's ring buffer (list capped at ``max_size``)."""

    __slots__ = ("entries", "max_size")

    def __init__(self, max_size: int = RING_BUFFER_MAX) -> None:
        self.entries: list[dict] = []
        self.max_size = max_size

    def extend(self, new_entries: list[dict]) -> None:
        self.entries.extend(new_entries)
        if len(self.entries) > self.max_size:
            # Keep only the most recent ``max_size`` samples.
            del self.entries[: len(self.entries) - self.max_size]

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)

    def latest_score(self) -> float:
        """Most recent non-None score, or 0.0 if none."""
        for e in reversed(self.entries):
            s = e.get("score")
            if s is not None:
                return float(s)
        return 0.0

    def latest_raw(self) -> float | None:
        if not self.entries:
            return None
        return self.entries[-1].get("raw")

    def slice_block(self, block_size: int) -> list[dict]:
        """Return up to the last ``block_size`` entries (newest first kept
        in chronological order)."""
        if len(self.entries) > block_size:
            return self.entries[-block_size:]
        return self.entries


class RingBuffer:
    """Thread-safe multi-channel real-time buffer.

    Drop-in replacement for the legacy module-level ``ring_buffers`` dict +
    ``buffer_lock``.  Public methods mirror the read/write/clear access
    patterns that ``server.py``'s ``api_poll`` / ``api_reset`` used.
    """

    def __init__(self, max_size: int = RING_BUFFER_MAX) -> None:
        self._channels: dict[str, ChannelStore] = {}
        self._lock = threading.Lock()
        self._max_size = max_size

    # -- write --------------------------------------------------------------

    def ingest(self, channel_entries: dict[str, list[dict]]) -> None:
        """Append new entries per channel.  ``channel_entries`` shape::

            {"C-1": [ {raw, score, received_at, channel}, ... ], ...}
        """
        with self._lock:
            for ch, entries in channel_entries.items():
                store = self._channels.get(ch)
                if store is None:
                    store = ChannelStore(self._max_size)
                    self._channels[ch] = store
                store.extend(entries)

    def clear(self) -> None:
        with self._lock:
            self._channels.clear()

    # -- read ---------------------------------------------------------------

    def channels(self) -> list[str]:
        with self._lock:
            return list(self._channels.keys())

    def total_points(self) -> int:
        with self._lock:
            return sum(len(s) for s in self._channels.values())

    def snapshot_block(self, block_size: int) -> dict[str, dict]:
        """Return per-channel latest block in chart-ready form::

            {ch: {"telemetry": [[ts_ms, raw], ...],
                  "scores":    [[ts_ms, score_or_0], ...]}}

        Shape is identical to the legacy ``api_poll`` ``channels`` payload.
        """
        out: dict[str, dict] = {}
        with self._lock:
            for ch, store in self._channels.items():
                slice_buf = store.slice_block(block_size)
                tele = [
                    [int(e["received_at"] * 1000), e["raw"]] for e in slice_buf
                ]
                sc = [
                    [
                        int(e["received_at"] * 1000),
                        e["score"] if e.get("score") is not None else 0.0,
                    ]
                    for e in slice_buf
                ]
                out[ch] = {"telemetry": tele, "scores": sc}
        return out

    def raw_block_entries(self, channel: str, block_size: int) -> list[dict]:
        """Raw entry dicts (no chart reshape) — used by health/alert
        services that need the original ``raw``/``score`` arrays."""
        with self._lock:
            store = self._channels.get(channel)
            if store is None:
                return []
            return list(store.slice_block(block_size))

    def raw_block_entries_aligned(
        self, channels: list[str], block_size: int
    ) -> dict[str, list[dict]]:
        """Return multi-channel aligned block_size entries.

        Takes the latest ``block_size`` entries per channel and truncates
        all to the shortest channel's length so array indices are aligned
        across channels.  Used by the joint detector to stack sibling
        channels' scores into a (T, n_channels) matrix for co-anomaly
        consensus.

        Returns an empty dict if any requested channel is missing — the
        caller should skip joint detection in that case.
        """
        with self._lock:
            per_channel: dict[str, list[dict]] = {}
            min_len = block_size
            for ch in channels:
                store = self._channels.get(ch)
                if store is None:
                    return {}  # missing channel → caller skips
                entries = store.slice_block(block_size)
                per_channel[ch] = entries
                if len(entries) < min_len:
                    min_len = len(entries)
            if min_len == 0:
                return {}
            # Truncate all to the shortest (keeps the most-recent tail).
            return {ch: entries[-min_len:] for ch, entries in per_channel.items()}

    def all_channel_scores(self, block_size: int) -> dict[str, list[float]]:
        """Per-channel score arrays (length ≤ block_size).  Used by the
        health service to compute the channel/system health values."""
        with self._lock:
            out: dict[str, list[float]] = {}
            for ch, store in self._channels.items():
                buf = store.slice_block(block_size)
                out[ch] = [
                    float(e["score"]) if e.get("score") is not None else 0.0
                    for e in buf
                ]
            return out

    def latest_metrics(self) -> dict[str, dict]:
        """Per-channel latest raw/score snapshot (for dashboard cards)."""
        with self._lock:
            out: dict[str, dict] = {}
            for ch, store in self._channels.items():
                out[ch] = {
                    "raw": store.latest_raw(),
                    "score": store.latest_score(),
                    "points": len(store),
                    "received_at": (
                        store.entries[-1]["received_at"] if store.entries else None
                    ),
                }
            return out


__all__ = ["RingBuffer", "ChannelStore"]
