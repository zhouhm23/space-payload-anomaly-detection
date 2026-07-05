"""PHM layer unit tests — fast, no model loading.

Covers:
  - health formula correctness
  - warning lifecycle (pending → confirmed/false)
  - ring buffer slicing + sizing
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from phm.services.health_service import channel_health
from phm.database import RingBuffer
from phm.database.warning_store import WarningStore
from phm.database.alert_store import AlertStore


# ---------------------------------------------------------------------------
# Health formula
# ---------------------------------------------------------------------------
class TestChannelHealth:
    def test_all_normal(self):
        assert channel_health([0.1, 0.2, 0.3, 0.4]) == 100.0

    def test_all_anomalous(self):
        assert channel_health([0.8, 0.9, 1.0]) == 0.0

    def test_half_half(self):
        # threshold 0.7 → 2 of 4 normal
        assert channel_health([0.1, 0.2, 0.8, 0.9]) == 50.0

    def test_empty_returns_100(self):
        assert channel_health([]) == 100.0

    def test_boundary_exactly_threshold_is_normal(self):
        # score == threshold is considered normal (≤)
        assert channel_health([0.7, 0.7]) == 100.0


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------
class TestRingBuffer:
    def _entry(self, ch, raw, score, ts):
        return {"raw": raw, "score": score, "received_at": ts, "channel": ch}

    def test_ingest_and_slice(self):
        rb = RingBuffer(max_size=10)
        rb.ingest({"C-1": [self._entry("C-1", 0.1, 0.2, t) for t in range(5)]})
        snap = rb.snapshot_block(3)
        assert "C-1" in snap
        assert len(snap["C-1"]["telemetry"]) == 3
        # slice(3) keeps ts=2,3,4 (last 3 of 0..4) → last is ts=4 → 4000 ms
        assert snap["C-1"]["telemetry"][-1] == [4_000, 0.1]

    def test_cap(self):
        rb = RingBuffer(max_size=3)
        rb.ingest({"C-1": [self._entry("C-1", float(i), 0.0, i) for i in range(5)]})
        assert rb.total_points() == 3  # capped

    def test_clear(self):
        rb = RingBuffer()
        rb.ingest({"C-1": [self._entry("C-1", 1.0, 0.0, 0.0)]})
        rb.clear()
        assert rb.total_points() == 0
        assert rb.channels() == []

    def test_score_none_becomes_zero(self):
        rb = RingBuffer()
        rb.ingest({"C-1": [{"raw": 1.0, "score": None, "received_at": 0.0, "channel": "C-1"}]})
        snap = rb.snapshot_block(10)
        assert snap["C-1"]["scores"][0][1] == 0.0

    def test_all_channel_scores(self):
        rb = RingBuffer()
        rb.ingest({
            "C-1": [self._entry("C-1", 0.0, 0.5, 0.0)],
            "C-2": [self._entry("C-2", 0.0, 0.9, 0.0)],
        })
        scores = rb.all_channel_scores(10)
        assert scores["C-1"] == [0.5]
        assert scores["C-2"] == [0.9]


# ---------------------------------------------------------------------------
# Warning lifecycle
# ---------------------------------------------------------------------------
class TestWarningStore:
    def test_pending_then_confirmed(self):
        ws = WarningStore()
        ws.add_pending("C-1", predict_start=0.0, predict_end=1.0, max_predict_score=0.9)
        # Measured data in window exceeds threshold → confirmed
        changed = ws.verify("C-1", [(0.5, 0.8)])
        assert changed == 1
        items = ws.all()
        assert items[0]["status"] == "confirmed"

    def test_pending_then_false(self):
        ws = WarningStore()
        ws.add_pending("C-1", predict_start=0.0, predict_end=1.0, max_predict_score=0.9)
        changed = ws.verify("C-1", [(0.5, 0.3)])
        assert changed == 1
        assert ws.all()[0]["status"] == "false"

    def test_dedupe_overlapping(self):
        ws = WarningStore()
        first = ws.add_pending("C-1", 0.0, 1.0, 0.9)
        second = ws.add_pending("C-1", 0.5, 1.5, 0.95)  # overlaps → skipped
        assert first is not None
        assert second is None
        assert len(ws.all()) == 1

    def test_verify_skips_future_window(self):
        ws = WarningStore()
        # Use a far-future window so ``now < predict_end`` holds.
        import time as _t
        far_future = _t.time() + 10_000.0
        ws.add_pending("C-1", predict_start=far_future - 100.0, predict_end=far_future, max_predict_score=0.9)
        changed = ws.verify("C-1", [(far_future - 50.0, 0.99)])
        assert changed == 0
