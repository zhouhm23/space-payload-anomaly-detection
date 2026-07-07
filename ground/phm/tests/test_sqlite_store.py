"""Unit tests for SQLiteStore.

Tests schema creation, batch insertion, query methods and lifecycle
(start / close) using a temporary database file.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from phm.database.sqlite_store import SQLiteStore
from phm.algorithm.cascade_types import LayerResult, CascadeOutput
from phm.algorithm.cascade_types import (
    LAYER_L1_CLASSIC, LAYER_L2_DL, LAYER_L3_PHYSICAL,
    DECISION_PASS, DECISION_ALERT,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_phm.db"


@pytest.fixture
def store(db_path):
    s = SQLiteStore(db_path, batch_size=5, flush_interval=0.5, enabled=True)
    s.start()
    yield s
    s.close()


class TestSQLiteStore:

    def test_init_creates_tables(self, db_path):
        """init() should create all three tables."""
        s = SQLiteStore(db_path, enabled=True)
        # Tables should exist even before start()
        tables = s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t[0] for t in tables}
        assert "raw_telemetry" in names
        assert "detection_results" in names
        assert "alert_records" in names
        s.close()

    def test_idempotent_schema(self, db_path):
        """Re-init should not error (CREATE IF NOT EXISTS)."""
        s1 = SQLiteStore(db_path, enabled=True)
        s1.close()
        # Second init on same file should succeed
        s2 = SQLiteStore(db_path, enabled=True)
        s2.close()

    def test_enqueue_and_query_telemetry(self, store):
        """Enqueued telemetry should appear in query_history."""
        for i in range(10):
            store.enqueue_telemetry("C-1", float(i), float(i) * 0.1, time.time() - (10 - i))
        # Wait for flush
        time.sleep(1.0)
        rows = store.query_history("C-1", limit=100)
        assert len(rows) == 10
        assert rows[0]["raw"] == 0.0
        assert rows[-1]["raw"] == 9.0

    def test_query_history_time_filter(self, store):
        """query_history should filter by start/end time."""
        t0 = time.time()
        for i in range(5):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
        time.sleep(1.0)
        rows = store.query_history("C-1", start_time=t0 + 2, end_time=t0 + 4)
        assert len(rows) == 3  # indices 2,3,4

    def test_enqueue_detection(self, store):
        """CascadeOutput should be persisted as a detection result."""
        cascade = CascadeOutput(
            channel="C-1",
            final_scores=np.array([0.1, 0.5, 0.9], dtype=np.float32),
            layers=[
                LayerResult(LAYER_L1_CLASSIC, DECISION_PASS, 0.0, {"rules": []}),
                LayerResult(LAYER_L2_DL, DECISION_PASS, 0.9, {"model": "mock"}),
                LayerResult(LAYER_L3_PHYSICAL, DECISION_ALERT, 0.95,
                            {"rules": ["range_boundary"]}),
            ],
        )
        store.enqueue_detection("C-1", time.time(), cascade)
        time.sleep(1.0)
        results = store.query_detection("C-1")
        assert len(results) == 1
        r = results[0]
        assert r["channel"] == "C-1"
        assert r["l1_decision"] == "pass"
        assert r["final_score"] == pytest.approx(0.9, abs=0.01)
        assert "range_boundary" in r["l3_rules"]

    def test_enqueue_alert(self, store):
        """Alert dict should be persisted."""
        store.enqueue_alert({
            "channel": "C-1",
            "type": "measured",
            "score": 0.85,
            "message": "test alert",
            "time": time.time(),
        })
        time.sleep(1.0)
        alerts = store.query_alerts()
        assert len(alerts) == 1
        assert alerts[0]["channel"] == "C-1"
        assert alerts[0]["alert_type"] == "measured"

    def test_disabled_store_noop(self, tmp_path):
        """When enabled=False, all operations should be no-ops."""
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        s.start()
        s.enqueue_telemetry("C-1", 1.0, 0.5, time.time())
        assert s.query_history() == []
        assert s.query_detection() == []
        s.close()

    def test_stats(self, store):
        """stats() should return row counts."""
        store.enqueue_telemetry("C-1", 1.0, 0.5, time.time())
        store.enqueue_telemetry("C-1", 2.0, 0.6, time.time())
        time.sleep(1.0)
        stats = store.stats()
        assert stats["enabled"] is True
        assert stats["raw_telemetry"] >= 2
        assert "predictions" in stats

    def test_enqueue_and_query_predictions(self, store):
        """Predicted values + scores should be persisted and queryable."""
        t0 = time.time()
        store.enqueue_predictions(
            channel="C-1",
            origin_ts=t0,
            predict_start=t0 + 0.02,
            predict_end=t0 + 0.02 * 96,
            prediction=[1.0, 2.0, 3.0],
            predict_scores=[0.1, 0.2, 0.3],
            model="linear",
        )
        time.sleep(1.0)
        preds = store.query_predictions("C-1")
        assert len(preds) == 1
        p = preds[0]
        assert p["channel"] == "C-1"
        assert p["prediction"] == [1.0, 2.0, 3.0]
        assert p["predict_scores"] == [0.1, 0.2, 0.3]
        assert p["model"] == "linear"

    def test_query_window_latest(self, store):
        """query_window with end_ts=None should return latest N points."""
        t0 = time.time()
        for i in range(10):
            store.enqueue_telemetry("C-1", float(i), float(i) * 0.1, t0 + i * 0.1)
        time.sleep(1.0)
        # Ask for latest 5
        w = store.query_window("C-1", count=5)
        assert w["count"] == 5
        assert w["raw"][0]["raw"] == 5.0  # 5th-from-last
        assert w["raw"][-1]["raw"] == 9.0  # latest
        # end_ts should be set to the latest point
        assert w["end_ts"] is not None

    def test_query_window_with_end_ts(self, store):
        """query_window with explicit end_ts should respect the right edge."""
        t0 = time.time()
        for i in range(10):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i * 0.1)
        time.sleep(1.0)
        # Right edge = t0 + 5*0.1 (point index 5, value 5.0)
        mid_ts = t0 + 5 * 0.1
        w = store.query_window("C-1", count=10, end_ts=mid_ts)
        assert w["count"] == 6  # indices 0..5
        assert w["raw"][-1]["raw"] == 5.0

    def test_query_window_includes_predictions(self, store):
        """query_window should include predictions within the window."""
        t0 = time.time()
        for i in range(10):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i * 0.1)
        # Prediction with origin at the last point
        store.enqueue_predictions(
            channel="C-1",
            origin_ts=t0 + 9 * 0.1,
            predict_start=t0 + 9 * 0.1 + 0.1,
            predict_end=t0 + 9 * 0.1 + 0.1 * 96,
            prediction=[10.0, 11.0],
            predict_scores=[0.5, 0.6],
            model="linear",
        )
        time.sleep(1.0)
        w = store.query_window("C-1", count=10)
        assert w["count"] == 10
        assert len(w["predictions"]) == 1
        assert w["predictions"][0]["prediction"] == [10.0, 11.0]

    def test_query_window_empty_channel(self, store):
        """query_window on non-existent channel should return empty."""
        w = store.query_window("NOPE", count=100)
        assert w["count"] == 0
        assert w["raw"] == []
        assert w["predictions"] == []

    def test_query_window_dedup_overlapping(self, store):
        """query_window should deduplicate overlapping timestamps from
        auto-poll cycles that produce overlapping blocks."""
        t0 = time.time()
        # First poll: points 0..9
        for i in range(10):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i * 0.1)
        # Second poll: points 5..9 again (overlap) + 10..14 (new)
        for i in range(5, 15):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i * 0.1)
        time.sleep(1.0)
        # Request 20 points but only 15 unique timestamps exist
        w = store.query_window("C-1", count=20)
        assert w["count"] == 15  # deduped, not 20
        # Verify strictly ascending timestamps
        ts_list = [p["received_at"] for p in w["raw"]]
        for i in range(1, len(ts_list)):
            assert ts_list[i] > ts_list[i - 1], f"Non-ascending at index {i}"

    def test_close_drains_queue(self, db_path):
        """close() should flush remaining items before closing."""
        s = SQLiteStore(db_path, batch_size=1000, flush_interval=100, enabled=True)
        s.start()
        # Enqueue items that won't trigger a batch flush (batch_size=1000)
        for i in range(10):
            s.enqueue_telemetry("C-1", float(i), 0.1, time.time())
        # close() should drain
        s.close()
        # Re-open and verify
        s2 = SQLiteStore(db_path, enabled=True)
        rows = s2.query_history("C-1")
        assert len(rows) == 10
        s2.close()

    def test_thread_safety(self, store):
        """Concurrent enqueues from multiple threads should not lose data."""
        import threading

        def writer(thread_id):
            for i in range(20):
                store.enqueue_telemetry(
                    f"ch-{thread_id}", float(i), 0.1, time.time()
                )

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(1.5)
        total = 0
        for t in range(4):
            total += len(store.query_history(f"ch-{t}"))
        assert total == 80  # 4 threads × 20 items
