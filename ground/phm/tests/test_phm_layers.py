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
        # threshold 0.5 → 2 of 4 normal
        assert channel_health([0.1, 0.2, 0.8, 0.9]) == 50.0

    def test_empty_returns_100(self):
        assert channel_health([]) == 100.0

    def test_boundary_exactly_threshold_is_normal(self):
        # score == threshold is considered normal (≤)
        assert channel_health([0.5, 0.5]) == 100.0


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
        import time as _t
        far_future = _t.time() + 10_000.0
        ws.add_pending("C-1", predict_start=far_future - 100.0, predict_end=far_future, max_predict_score=0.9)
        changed = ws.verify("C-1", [(far_future - 50.0, 0.99)])
        assert changed == 0


# ---------------------------------------------------------------------------
# HealthService folder aggregation (Slice 0)
# ---------------------------------------------------------------------------
class _FakeConfigService:
    """Minimal stand-in for ConfigService — serves a fixed config dict."""

    def __init__(self, config: dict):
        self._config = config

    def load(self) -> dict:
        return self._config


class TestHealthServiceFolderAggregation:
    """Verify HealthService rolls per-channel health up to folders."""

    def _entry(self, ch, raw, score, ts):
        return {"raw": raw, "score": score, "received_at": ts, "channel": ch}

    def _tree(self):
        """3 sensors: 2 in folder_A, 1 orphan (no folder)."""
        return [
            {
                "id": "folder_A",
                "name": "电源模块",
                "type": "folder",
                "children": [
                    {"id": "s1", "type": "sensor", "channelName": "C-1", "sourceId": "file:X/C-1"},
                    {"id": "s2", "type": "sensor", "channelName": "C-2", "sourceId": "file:X/C-2"},
                ],
            },
            {"id": "s3", "type": "sensor", "channelName": "C-3", "sourceId": "file:X/C-3"},
        ]

    def _ring_with_healths(self, c1_scores, c2_scores, c3_scores):
        rb = RingBuffer()
        rb.ingest({"C-1": [self._entry("C-1", 0.0, s, 0.0) for s in c1_scores]})
        rb.ingest({"C-2": [self._entry("C-2", 0.0, s, 0.0) for s in c2_scores]})
        rb.ingest({"C-3": [self._entry("C-3", 0.0, s, 0.0) for s in c3_scores]})
        return rb

    def test_min_strategy_worst_sensor_wins(self):
        # C-1 = 100% normal, C-2 = 0% normal → folder min should be 0.0
        rb = self._ring_with_healths([0.1, 0.2], [0.8, 0.9], [0.5, 0.5])
        cfg = _FakeConfigService({"device_tree": self._tree(), "aggregation_strategy": "min"})
        from phm.services.health_service import HealthService

        result = HealthService(rb, cfg).system_health()
        assert "folders" in result
        assert result["folders"]["folder_A"]["health"] == 0.0
        assert result["folders"]["folder_A"]["strategy"] == "min"
        assert set(result["folders"]["folder_A"]["channels"]) == {"C-1", "C-2"}

    def test_mean_strategy_averages(self):
        # C-1 = 100% normal, C-2 = 0% normal → folder mean = 50.0
        rb = self._ring_with_healths([0.1, 0.2], [0.8, 0.9], [0.5, 0.5])
        cfg = _FakeConfigService({"device_tree": self._tree(), "aggregation_strategy": "mean"})
        from phm.services.health_service import HealthService

        result = HealthService(rb, cfg).system_health()
        assert result["folders"]["folder_A"]["health"] == 50.0
        assert result["folders"]["folder_A"]["strategy"] == "mean"

    def test_orphan_sensor_excluded_from_folders(self):
        rb = self._ring_with_healths([0.1], [0.2], [0.9])
        cfg = _FakeConfigService({"device_tree": self._tree(), "aggregation_strategy": "min"})
        from phm.services.health_service import HealthService

        result = HealthService(rb, cfg).system_health()
        # C-3 is an orphan — it appears in channels but NOT in any folder entry
        assert "C-3" in result["channels"]
        assert "folder_A" in result["folders"]
        assert "C-3" not in result["folders"]["folder_A"]["channels"]

    def test_system_health_unaffected_by_aggregation(self):
        rb = self._ring_with_healths([0.1], [0.8], [0.5])
        cfg = _FakeConfigService({"device_tree": self._tree(), "aggregation_strategy": "min"})
        from phm.services.health_service import HealthService

        with_cfg = HealthService(rb, cfg).system_health()
        without_cfg = HealthService(rb, None).system_health()
        # system / channels are identical regardless of folder aggregation
        assert with_cfg["system"] == without_cfg["system"]
        assert with_cfg["channels"] == without_cfg["channels"]
        # only difference is the folders key
        assert "folders" in with_cfg
        assert "folders" not in without_cfg

    def test_folder_with_no_data_is_skipped(self):
        # C-1, C-2 have data; a second folder references C-4 which has no data
        rb = self._ring_with_healths([0.1], [0.2], [])
        tree = self._tree() + [
            {"id": "folder_B", "name": "空模块", "type": "folder",
             "children": [{"id": "s4", "type": "sensor", "channelName": "C-4", "sourceId": "file:X/C-4"}]}
        ]
        cfg = _FakeConfigService({"device_tree": tree, "aggregation_strategy": "min"})
        from phm.services.health_service import HealthService

        result = HealthService(rb, cfg).system_health()
        assert "folder_A" in result["folders"]
        assert "folder_B" not in result["folders"]  # no data → skipped

    def test_default_strategy_is_min_when_key_missing(self):
        rb = self._ring_with_healths([0.1], [0.9], [0.5])
        cfg = _FakeConfigService({"device_tree": self._tree()})  # no aggregation_strategy key
        from phm.services.health_service import HealthService

        result = HealthService(rb, cfg).system_health()
        assert result["folders"]["folder_A"]["strategy"] == "min"
        assert result["folders"]["folder_A"]["health"] == 0.0  # C-2 is all-anomalous
