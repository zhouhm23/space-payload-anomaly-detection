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
