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
        """init() should create the fixed tables.

        Per-channel telemetry_* tables are created on demand, so only the
        fixed detection_results / alert_records tables must exist at init.
        """
        s = SQLiteStore(db_path, enabled=True)
        tables = s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t[0] for t in tables}
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
        t0 = time.time()
        store.enqueue_telemetry("C-1", 1.0, 0.5, t0)
        store.enqueue_telemetry("C-1", 2.0, 0.6, t0 + 1)
        time.sleep(1.0)
        stats = store.stats()
        assert stats["enabled"] is True
        assert stats["telemetry"] >= 2
        # Channel names in stats are derived from table names (sanitised:
        # non-alphanumeric → underscore), so "C-1" appears as "C_1".
        assert "C_1" in stats["telemetry_by_channel"]

    def test_enqueue_and_query_predictions(self, store):
        """Predicted values + scores should be persisted and queryable.

        Predictions are now individual UPSERT rows in the unified telemetry
        table (one row per predicted point), so query via query_window.
        """
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
        w = store.query_window("C-1", count=100)
        preds = [d for d in w["data"] if d["predicted_value"] is not None]
        assert len(preds) == 3
        assert preds[0]["predicted_value"] == 1.0
        assert preds[0]["predicted_anomaly_score"] == pytest.approx(0.1)

    def test_query_window_latest(self, store):
        """query_window with end_ts=None should return latest N points."""
        t0 = time.time()
        for i in range(10):
            store.enqueue_telemetry("C-1", float(i), float(i) * 0.1, t0 + i * 0.1)
        time.sleep(1.0)
        # Ask for latest 5
        w = store.query_window("C-1", count=5)
        assert w["count"] == 5
        assert w["data"][0]["raw_value"] == 5.0  # 5th-from-last
        assert w["data"][-1]["raw_value"] == 9.0  # latest
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
        assert w["data"][-1]["raw_value"] == 5.0

    def test_query_window_includes_predictions(self, store):
        """query_window should include predictions within the window.

        Predictions are now merged into the unified data rows via UPSERT,
        so check predicted_value on the data entries.
        """
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
        w = store.query_window("C-1", count=100)
        preds = [d for d in w["data"] if d["predicted_value"] is not None]
        assert len(preds) == 2
        assert preds[0]["predicted_value"] == 10.0

    def test_query_window_empty_channel(self, store):
        """query_window on non-existent channel should return empty."""
        w = store.query_window("NOPE", count=100)
        assert w["count"] == 0
        assert w["data"] == []

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
        ts_list = [d["timestamp"] for d in w["data"]]
        for i in range(1, len(ts_list)):
            assert ts_list[i] > ts_list[i - 1], f"Non-ascending at index {i}"

    def test_close_drains_queue(self, db_path):
        """close() should flush remaining items before closing."""
        s = SQLiteStore(db_path, batch_size=1000, flush_interval=100, enabled=True)
        s.start()
        # Enqueue items that won't trigger a batch flush (batch_size=1000).
        # Use strictly increasing timestamps so they don't quantise to the
        # same PRIMARY KEY (quantum = 1/sample_rate = 0.01s).
        t0 = time.time()
        for i in range(10):
            s.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
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

        # Each thread uses a distinct timestamp base so points don't
        # quantise to the same PRIMARY KEY (quantum = 0.01s).
        t0 = time.time()

        def writer(thread_id):
            for i in range(20):
                store.enqueue_telemetry(
                    f"ch-{thread_id}", float(i), 0.1,
                    t0 + thread_id * 100 + i,
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


class TestStoreMutations:
    """Tests for the synchronous mutation API (delete / update).

    Added for the database-panel feature: the panel needs DELETE on
    ``raw_telemetry`` / ``detection_results`` and PATCH on
    ``alert_records.status``.  These run synchronously against the same
    connection the flush thread writes to, so they must hold
    ``_write_lock`` (covered indirectly by asserting no rows are lost and
    no exception is raised while the background thread is running).
    """

    # ---- query_alerts now returns id ----

    def test_query_alerts_includes_id(self, store):
        """query_alerts should include the row id (needed for PATCH)."""
        store.enqueue_alert({
            "channel": "C-1", "type": "measured", "score": 0.8,
            "message": "has id", "time": time.time(),
        })
        time.sleep(1.0)
        alerts = store.query_alerts()
        assert len(alerts) == 1
        assert alerts[0]["id"] == 1  # AUTOINCREMENT starts at 1

    # ---- delete_history ----

    def test_delete_history_by_channel(self, store):
        """delete_history(channel=...) should remove only that channel."""
        t0 = time.time()
        for i in range(5):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
            store.enqueue_telemetry("C-2", float(i), 0.1, t0 + i)
        time.sleep(1.0)
        deleted = store.delete_history(channel="C-1")
        assert deleted == 5
        assert store.query_history("C-1") == []
        assert len(store.query_history("C-2")) == 5

    def test_delete_history_by_time_range(self, store):
        """delete_history(start, end) should remove only rows in range."""
        t0 = time.time()
        for i in range(10):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
        time.sleep(1.0)
        # delete the middle 4 points (indices 3..6)
        deleted = store.delete_history(
            channel="C-1", start_time=t0 + 3, end_time=t0 + 6,
        )
        assert deleted == 4
        remaining = store.query_history("C-1")
        assert len(remaining) == 6  # 0,1,2,7,8,9
        times = [r["received_at"] for r in remaining]
        assert all(t < t0 + 3 or t > t0 + 6 for t in times)

    def test_delete_history_all_requires_no_filter(self, store):
        """delete_history() with no filter clears the whole table."""
        t0 = time.time()
        for i in range(3):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
        time.sleep(1.0)
        deleted = store.delete_history()
        assert deleted == 3
        assert store.query_history() == []

    def test_delete_history_disabled_store(self, tmp_path):
        """delete_history on a disabled store should return 0."""
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        assert s.delete_history(channel="C-1") == 0
        s.close()

    # ---- delete_detection ----

    def test_delete_detection_by_channel(self, store):
        """delete_detection(channel=...) should remove only that channel.

        Note the time column is ``timestamp`` (not ``received_at``).
        """
        cascade = CascadeOutput(
            channel="C-1",
            final_scores=np.array([0.1, 0.5], dtype=np.float32),
            layers=[LayerResult(LAYER_L1_CLASSIC, DECISION_PASS, 0.0, {})],
        )
        store.enqueue_detection("C-1", time.time(), cascade)
        store.enqueue_detection("C-2", time.time(), cascade)
        time.sleep(1.0)
        assert len(store.query_detection("C-1")) == 1
        assert len(store.query_detection("C-2")) == 1
        deleted = store.delete_detection(channel="C-1")
        assert deleted == 1
        assert store.query_detection("C-1") == []
        assert len(store.query_detection("C-2")) == 1

    def test_delete_detection_by_time(self, store):
        """delete_detection(start, end) should filter on the timestamp column."""
        t0 = time.time()
        cascade = CascadeOutput(
            channel="C-1",
            final_scores=np.array([0.1], dtype=np.float32),
            layers=[LayerResult(LAYER_L1_CLASSIC, DECISION_PASS, 0.0, {})],
        )
        for i in range(5):
            store.enqueue_detection("C-1", t0 + i, cascade)
        time.sleep(1.0)
        deleted = store.delete_detection(
            channel="C-1", start_time=t0 + 1, end_time=t0 + 3,
        )
        assert deleted == 3
        assert len(store.query_detection("C-1")) == 2  # t0+0 and t0+4 remain

    def test_delete_detection_disabled_store(self, tmp_path):
        """delete_detection on a disabled store should return 0."""
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        assert s.delete_detection(channel="C-1") == 0
        s.close()

    # ---- update_alert_status ----

    def test_update_alert_status_pending_to_confirmed(self, store):
        """PATCH should flip pending → confirmed and set verified_at."""
        store.enqueue_alert({
            "channel": "C-1", "type": "predicted", "score": 0.85,
            "message": "pending warn", "time": time.time(), "status": "pending",
        })
        time.sleep(1.0)
        alerts = store.query_alerts()
        assert alerts[0]["status"] == "pending"
        aid = alerts[0]["id"]
        ok = store.update_alert_status(aid, "confirmed")
        assert ok is True
        updated = store.query_alerts()[0]
        assert updated["status"] == "confirmed"
        assert updated["verified_at"] is not None

    def test_update_alert_status_to_false(self, store):
        """PATCH should allow pending → false (false alarm)."""
        store.enqueue_alert({
            "channel": "C-1", "type": "predicted", "score": 0.8,
            "message": "false alarm", "time": time.time(), "status": "pending",
        })
        time.sleep(1.0)
        aid = store.query_alerts()[0]["id"]
        assert store.update_alert_status(aid, "false") is True
        assert store.query_alerts()[0]["status"] == "false"

    def test_update_alert_status_rejects_active(self, store):
        """``active`` is not in the patchable set — should return False."""
        store.enqueue_alert({
            "channel": "C-1", "type": "measured", "score": 0.9,
            "message": "active alert", "time": time.time(),
        })
        time.sleep(1.0)
        aid = store.query_alerts()[0]["id"]
        # Default status for measured alerts is 'active'; PATCH to 'active'
        # should be refused (status not in allow-list).
        assert store.update_alert_status(aid, "active") is False
        assert store.query_alerts()[0]["status"] == "active"  # unchanged

    def test_update_alert_status_rejects_bogus(self, store):
        """An unknown status string should return False, not write."""
        store.enqueue_alert({
            "channel": "C-1", "type": "measured", "score": 0.9,
            "message": "bogus", "time": time.time(), "status": "pending",
        })
        time.sleep(1.0)
        aid = store.query_alerts()[0]["id"]
        assert store.update_alert_status(aid, "bogus") is False
        assert store.query_alerts()[0]["status"] == "pending"  # unchanged

    def test_update_alert_status_missing_id(self, store):
        """A non-existent id should return False."""
        assert store.update_alert_status(99999, "confirmed") is False

    def test_update_alert_status_disabled_store(self, tmp_path):
        """update_alert_status on a disabled store should return False."""
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        assert s.update_alert_status(1, "confirmed") is False
        s.close()


class TestAlertVerdictColumns:
    """Tests for llm_verdict / human_verdict / final_status on alert_records."""

    def test_query_alerts_includes_verdict_fields(self, store):
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.9, "time": 1000.0})
        time.sleep(1.0)
        alerts = store.query_alerts()
        assert "llm_verdict" in alerts[0]
        assert "human_verdict" in alerts[0]
        assert "final_status" in alerts[0]

    def test_update_alert_verdict_human(self, store):
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.9, "time": 1000.0})
        time.sleep(1.0)
        ok = store.update_alert_verdict("C-1", 1000.0, "real")
        assert ok is True
        alerts = store.query_alerts()
        assert alerts[0]["human_verdict"] == "real"
        assert alerts[0]["final_status"] == "real"

    def test_update_alert_verdict_llm(self, store):
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.9, "time": 1000.0})
        time.sleep(1.0)
        ok = store.update_alert_verdict("C-1", 1000.0, "false_alarm", is_llm=True)
        assert ok is True
        alerts = store.query_alerts()
        assert alerts[0]["llm_verdict"] == "false_alarm"
        assert alerts[0]["final_status"] == "false_alarm"

    def test_final_status_llm_when_no_human(self, store):
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.9, "time": 1000.0})
        time.sleep(1.0)
        store._conn.execute(
            "UPDATE alert_records SET llm_verdict='false_alarm' WHERE channel='C-1' AND created_at=1000.0"
        )
        store._conn.commit()
        alerts = store.query_alerts()
        assert alerts[0]["final_status"] == "false_alarm"

    def test_final_status_falls_back_to_status(self, store):
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.9, "time": 1000.0})
        time.sleep(1.0)
        alerts = store.query_alerts()
        # No verdicts set → final_status == status ('active' for measured)
        assert alerts[0]["final_status"] == alerts[0]["status"]

    def test_update_alert_verdict_missing_row(self, store):
        ok = store.update_alert_verdict("C-1", 99999.0, "real")
        assert ok is False

    def test_update_alert_verdict_invalid_value(self, store):
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.9, "time": 1000.0})
        time.sleep(1.0)
        ok = store.update_alert_verdict("C-1", 1000.0, "bogus")
        assert ok is False

    def test_human_overrides_llm_in_final_status(self, store):
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.9, "time": 1000.0})
        time.sleep(1.0)
        store.update_alert_verdict("C-1", 1000.0, "false_alarm", is_llm=True)
        store.update_alert_verdict("C-1", 1000.0, "real")
        alerts = store.query_alerts()
        assert alerts[0]["llm_verdict"] == "false_alarm"
        assert alerts[0]["human_verdict"] == "real"
        assert alerts[0]["final_status"] == "real"


class TestDiagnosisVerdictColumn:
    """Tests for llm_verdict on diagnosis_records."""

    def test_save_diagnosis_with_verdict(self, store):
        store.save_diagnosis("C-1", "measured", 1000.0, "report", {}, 1.0, None, verdict="real")
        d = store.get_diagnosis("C-1", "measured", 1000.0)
        assert d["llm_verdict"] == "real"

    def test_save_diagnosis_without_verdict(self, store):
        store.save_diagnosis("C-1", "measured", 1000.0, "report", {}, 1.0, None)
        d = store.get_diagnosis("C-1", "measured", 1000.0)
        assert d["llm_verdict"] is None

    def test_list_diagnosis_keys_includes_verdict(self, store):
        store.save_diagnosis("C-1", "measured", 1000.0, "report", {}, 1.0, None, verdict="uncertain")
        keys = store.list_diagnosis_keys()
        assert keys[0]["llm_verdict"] == "uncertain"

    def test_migration_adds_verdict_columns(self, tmp_path):
        """Existing DB without verdict columns gets them added on open."""
        import sqlite3 as _sqlite3
        db = tmp_path / "old.db"
        conn = _sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE alert_records (id INTEGER PRIMARY KEY, channel TEXT, "
            "alert_type TEXT, score REAL, message TEXT, created_at REAL, "
            "status TEXT DEFAULT 'active', verified_at REAL, ingested_at REAL)"
        )
        conn.execute(
            "CREATE TABLE diagnosis_records (id INTEGER PRIMARY KEY, channel TEXT, "
            "alert_type TEXT, alert_ts REAL, diagnosis TEXT, context_summary TEXT, "
            "elapsed_sec REAL, error TEXT, created_at REAL)"
        )
        conn.commit()
        conn.close()
        s = SQLiteStore(db, batch_size=5, flush_interval=0.5, enabled=True)
        s.start()
        cols = s._conn.execute("PRAGMA table_info(alert_records)").fetchall()
        col_names = [c[1] for c in cols]
        assert "llm_verdict" in col_names
        assert "human_verdict" in col_names
        diag_cols = [c[1] for c in s._conn.execute("PRAGMA table_info(diagnosis_records)").fetchall()]
        assert "llm_verdict" in diag_cols
        s.close()


class TestSoftDelete:
    """Tests for the soft-delete feature (is_deleted / deleted_at)."""

    def test_schema_has_soft_delete_columns(self, store):
        """All three business tables should have is_deleted + deleted_at."""
        for table in ("detection_results", "alert_records", "diagnosis_records"):
            cols = [c[1] for c in store._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            assert "is_deleted" in cols, f"{table} missing is_deleted"
            assert "deleted_at" in cols, f"{table} missing deleted_at"

    def test_migration_adds_soft_delete_columns(self, tmp_path):
        """Pre-existing DB without soft-delete columns gets them on open."""
        import sqlite3 as _sqlite3
        db = tmp_path / "old_softdel.db"
        conn = _sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE alert_records (id INTEGER PRIMARY KEY, channel TEXT, "
            "alert_type TEXT, score REAL, message TEXT, created_at REAL, "
            "status TEXT DEFAULT 'active', verified_at REAL, "
            "llm_verdict TEXT, human_verdict TEXT, ingested_at REAL)"
        )
        conn.execute(
            "CREATE TABLE detection_results (id INTEGER PRIMARY KEY, channel TEXT, "
            "timestamp REAL, l1_decision TEXT, l1_score REAL, l1_detail TEXT, "
            "l2_score REAL, l3_score REAL, l3_rules TEXT, final_score REAL, ingested_at REAL)"
        )
        conn.execute(
            "CREATE TABLE diagnosis_records (id INTEGER PRIMARY KEY, channel TEXT, "
            "alert_type TEXT, alert_ts REAL, diagnosis TEXT, context_summary TEXT, "
            "elapsed_sec REAL, error TEXT, llm_verdict TEXT, created_at REAL)"
        )
        conn.commit()
        conn.close()
        s = SQLiteStore(db, batch_size=5, flush_interval=0.5, enabled=True)
        s.start()
        for table in ("alert_records", "detection_results", "diagnosis_records"):
            cols = [c[1] for c in s._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            assert "is_deleted" in cols
            assert "deleted_at" in cols
        s.close()

    def test_delete_alert_soft_deletes(self, store):
        """delete_alert marks rows is_deleted=1, query_alerts hides them."""
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.8, "time": 1000.0})
        store.enqueue_alert({"channel": "C-2", "type": "measured", "score": 0.9, "time": 2000.0})
        time.sleep(1.0)
        assert len(store.query_alerts()) == 2
        n = store.delete_alert(channel="C-1")
        assert n == 1
        alerts = store.query_alerts()
        assert len(alerts) == 1
        assert alerts[0]["channel"] == "C-2"

    def test_delete_alert_by_time_range(self, store):
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.8, "time": 1000.0})
        store.enqueue_alert({"channel": "C-2", "type": "measured", "score": 0.9, "time": 2000.0})
        time.sleep(1.0)
        n = store.delete_alert(start_time=1500.0)
        assert n == 1
        alerts = store.query_alerts()
        assert len(alerts) == 1
        assert alerts[0]["channel"] == "C-1"

    def test_delete_detection_soft_deletes(self, store):
        cascade = CascadeOutput(
            channel="C-1",
            final_scores=np.array([0.1], dtype=np.float32),
            layers=[],
        )
        store.enqueue_detection("C-1", 5000.0, cascade)
        time.sleep(1.0)
        assert len(store.query_detection()) == 1
        n = store.delete_detection(channel="C-1")
        assert n == 1
        assert store.query_detection() == []

    def test_delete_diagnosis_soft_deletes(self, store):
        store.save_diagnosis("C-1", "measured", 7000.0, "report", {}, 1.0, None, "real")
        assert store.get_diagnosis("C-1", "measured", 7000.0) is not None
        n = store.delete_diagnosis(channel="C-1")
        assert n == 1
        assert store.get_diagnosis("C-1", "measured", 7000.0) is None
        assert store.list_diagnosis_keys() == []

    def test_update_verdict_skips_deleted(self, store):
        """update_alert_verdict should not update soft-deleted rows."""
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.8, "time": 1000.0})
        time.sleep(1.0)
        store.delete_alert(channel="C-1")
        ok = store.update_alert_verdict("C-1", 1000.0, "real")
        assert ok is False

    def test_purge_deleted_alerts(self, store):
        """purge_deleted physically removes soft-deleted rows."""
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.8, "time": 1000.0})
        time.sleep(1.0)
        store.delete_alert(channel="C-1")
        n = store.purge_deleted("alert_records")
        assert n == 1
        # Verify physically gone
        count = store._conn.execute(
            "SELECT COUNT(*) FROM alert_records WHERE is_deleted = 1"
        ).fetchone()[0]
        assert count == 0

    def test_purge_deleted_invalid_table(self, store):
        """purge_deleted rejects unknown table names."""
        assert store.purge_deleted("bogus_table") == 0

    def test_stats_excludes_deleted(self, store):
        """stats() should only count non-deleted rows."""
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.8, "time": 1000.0})
        store.enqueue_alert({"channel": "C-2", "type": "measured", "score": 0.9, "time": 2000.0})
        time.sleep(1.0)
        store.delete_alert(channel="C-1")
        stats = store.stats()
        assert stats["alert_records"] == 1

    def test_delete_history_still_hard_deletes(self, store):
        """Telemetry tables use HARD delete (no soft-delete columns)."""
        t0 = time.time()
        for i in range(3):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
        time.sleep(1.0)
        n = store.delete_history(channel="C-1")
        assert n == 3
        # Verify physically removed
        assert store.query_history("C-1") == []


class TestAlertSnapshot:
    """Tests for raw_snapshot / score_snapshot on alert_records."""

    def test_schema_has_snapshot_columns(self, store):
        cols = [c[1] for c in store._conn.execute("PRAGMA table_info(alert_records)").fetchall()]
        assert "raw_snapshot" in cols
        assert "score_snapshot" in cols

    def test_enqueue_alert_with_snapshot_roundtrip(self, store):
        """enqueue_alert with raw_snapshot/score_snapshot → query_alerts returns them."""
        store.enqueue_alert({
            "channel": "C-1", "type": "measured", "score": 0.85,
            "time": 1000.0,
            "raw_snapshot": [0.1, 0.2, 0.3, 0.4, 0.5],
            "score_snapshot": [0.1, 0.5, 0.9, 0.3, 0.2],
        })
        time.sleep(1.0)
        alerts = store.query_alerts()
        assert len(alerts) == 1
        assert alerts[0]["raw_snapshot"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert alerts[0]["score_snapshot"] == [0.1, 0.5, 0.9, 0.3, 0.2]

    def test_enqueue_alert_without_snapshot(self, store):
        """Alerts without snapshots should have None (backward compat)."""
        store.enqueue_alert({"channel": "C-1", "type": "measured", "score": 0.5, "time": 1000.0})
        time.sleep(1.0)
        alerts = store.query_alerts()
        assert alerts[0]["raw_snapshot"] is None
        assert alerts[0]["score_snapshot"] is None

    def test_migration_adds_snapshot_columns(self, tmp_path):
        """Pre-existing DB without snapshot columns gets them on open."""
        import sqlite3 as _sqlite3
        db = tmp_path / "old_snap.db"
        conn = _sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE alert_records (id INTEGER PRIMARY KEY, channel TEXT, "
            "alert_type TEXT, score REAL, message TEXT, created_at REAL, "
            "status TEXT DEFAULT 'active', verified_at REAL, "
            "llm_verdict TEXT, human_verdict TEXT, ingested_at REAL, "
            "is_deleted INTEGER DEFAULT 0, deleted_at REAL)"
        )
        conn.commit()
        conn.close()
        s = SQLiteStore(db, batch_size=5, flush_interval=0.5, enabled=True)
        s.start()
        cols = [c[1] for c in s._conn.execute("PRAGMA table_info(alert_records)").fetchall()]
        assert "raw_snapshot" in cols
        assert "score_snapshot" in cols
        s.close()


# ─────────────────────────────────────────────────────────────────────────
# v1.2 telemetry data-layer extensions: pagination, count, soft-delete,
# manual insert, streaming export.
#
# Covers _migrate_tel_soft_delete_columns / query_tel_page / count_tel /
# soft_delete_tel / restore_tel / purge_tel / query_tel_deleted /
# insert_tel_manual / iter_tel_rows.
# ─────────────────────────────────────────────────────────────────────────


def _tel_table_name(channel: str) -> str:
    """Mirror SQLiteStore._tel_table so tests can hit the table directly."""
    safe = "".join(c if c.isalnum() else "_" for c in channel)
    return f"telemetry_{safe}"


def _insert_tel_row(store, channel, timestamp, raw_value=None,
                    anomaly_score=None, origin="acq", deleted_at=None):
    """Synchronously insert one telemetry row; returns the new rowid.

    Bypasses the async enqueue queue so tests are deterministic. Uses the
    full column set so origin / deleted_at can be set explicitly.
    """
    table = _tel_table_name(channel)
    store._ensure_tel_table(channel)
    cur = store._conn.execute(
        f'INSERT INTO "{table}" (timestamp, raw_value, anomaly_score, '
        f'ingested_at, origin, deleted_at) '
        f'VALUES (?, ?, ?, unixepoch(), ?, ?)',
        [float(timestamp), raw_value, anomaly_score, origin, deleted_at],
    )
    store._conn.commit()
    return cur.lastrowid


class TestTelSoftDeleteMigration:
    """W1: deleted_at + origin migration for telemetry_* tables."""

    def test_fresh_table_has_columns(self, store):
        """A freshly-created telemetry table must have deleted_at + origin
        (added via _ensure_tel_table's DDL, not the migration pass)."""
        store._ensure_tel_table("C-1")
        cols = [c[1] for c in store._conn.execute(
            'PRAGMA table_info("telemetry_C_1")').fetchall()]
        assert "deleted_at" in cols
        assert "origin" in cols

    def test_origin_default_is_acq(self, store):
        """A row inserted without specifying origin must get 'acq'."""
        _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0, origin="acq")
        page = store.query_tel_page("C-1", limit=10)
        assert page[0]["origin"] == "acq"

    def test_migration_idempotent_call_twice(self, store):
        """Calling _migrate_tel_soft_delete_columns twice must not raise
        'duplicate column' — the migration guards on column existence."""
        store._ensure_tel_table("C-1")
        # First migration (the table already has the columns from DDL).
        store._migrate_tel_soft_delete_columns()
        # Second migration — should be a no-op, no exception.
        store._migrate_tel_soft_delete_columns()
        cols = [c[1] for c in store._conn.execute(
            'PRAGMA table_info("telemetry_C_1")').fetchall()]
        assert cols.count("deleted_at") == 1
        assert cols.count("origin") == 1

    def test_migration_upgrades_old_db(self, tmp_path):
        """A pre-existing DB whose telemetry table predates the
        deleted_at/origin columns must get them added on open."""
        import sqlite3 as _sqlite3
        db = tmp_path / "old_tel.db"
        conn = _sqlite3.connect(str(db))
        # Old schema: no deleted_at, no origin.
        conn.execute(
            'CREATE TABLE telemetry_C_1 ('
            "timestamp REAL NOT NULL PRIMARY KEY, "
            "raw_value REAL, anomaly_score REAL, "
            "predicted_value REAL, predicted_anomaly_score REAL, "
            "origin_ts REAL, ingested_at REAL)"
        )
        # One pre-existing row.
        conn.execute(
            "INSERT INTO telemetry_C_1 (timestamp, raw_value, ingested_at) "
            "VALUES (1000.0, 1.0, 0)"
        )
        conn.commit()
        conn.close()

        s = SQLiteStore(db, batch_size=5, flush_interval=0.5, enabled=True)
        s.start()
        cols = [c[1] for c in s._conn.execute(
            'PRAGMA table_info("telemetry_C_1")').fetchall()]
        assert "deleted_at" in cols
        assert "origin" in cols
        # The pre-existing row should pick up origin='acq' (DEFAULT) and
        # deleted_at=NULL, so it is queryable as a live row.
        page = s.query_tel_page("C-1", limit=10)
        assert len(page) == 1
        assert page[0]["origin"] == "acq"
        assert page[0]["deleted_at"] is None
        s.close()


class TestQueryTelPage:
    """W2: paginated telemetry query."""

    def test_empty_channel_returns_empty(self, store):
        assert store.query_tel_page("NOPE", limit=10) == []

    def test_returns_newest_first(self, store):
        for i in range(5):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        page = store.query_tel_page("C-1", limit=10)
        assert len(page) == 5
        ts = [r["timestamp"] for r in page]
        assert ts == sorted(ts, reverse=True)  # DESC
        assert ts[0] == 1004.0

    def test_pagination_offset_limit(self, store):
        """With 100 rows, page(offset=20, limit=20) returns rows 21-40
        in newest-first order (timestamps 1004.0 down to ... )."""
        for i in range(100):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        page = store.query_tel_page("C-1", limit=20, offset=20)
        assert len(page) == 20
        # Newest-first: offset=20 skips the 20 newest (ts 1099..1080),
        # so this page covers ts 1079..1060.
        ts = [r["timestamp"] for r in page]
        assert ts == sorted(ts, reverse=True)
        assert ts[0] == 1079.0
        assert ts[-1] == 1060.0

    def test_offset_beyond_returns_empty(self, store):
        _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        assert store.query_tel_page("C-1", limit=10, offset=100) == []

    def test_time_range_filter(self, store):
        for i in range(10):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        page = store.query_tel_page(
            "C-1", limit=50, start_ts=1003.0, end_ts=1005.0)
        ts = [r["timestamp"] for r in page]
        assert all(1003.0 <= t <= 1005.0 for t in ts)
        assert len(page) == 3

    def test_excludes_deleted_by_default(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        _insert_tel_row(store, "C-1", 1001.0, raw_value=2.0)
        store.soft_delete_tel("C-1", [rid])
        page = store.query_tel_page("C-1", limit=10)
        assert len(page) == 1
        assert page[0]["timestamp"] == 1001.0

    def test_include_deleted_returns_all(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        _insert_tel_row(store, "C-1", 1001.0, raw_value=2.0)
        store.soft_delete_tel("C-1", [rid])
        page = store.query_tel_page("C-1", limit=10, include_deleted=True)
        assert len(page) == 2

    def test_row_dict_shape(self, store):
        """The returned dict must carry rowid as 'id' plus all columns."""
        _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0, anomaly_score=0.5)
        page = store.query_tel_page("C-1", limit=1)
        r = page[0]
        for key in ("id", "timestamp", "raw_value", "anomaly_score",
                    "predicted_value", "predicted_anomaly_score", "origin_ts",
                    "ingested_at", "deleted_at", "origin"):
            assert key in r, f"missing key {key}"
        assert isinstance(r["id"], int)

    def test_limit_clamped_to_max(self, store):
        """limit above 1000 is clamped; below 1 is clamped to 1."""
        _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        # Huge limit should not raise; returns the 1 available row.
        page = store.query_tel_page("C-1", limit=10000)
        assert len(page) == 1

    def test_disabled_store_returns_empty(self, tmp_path):
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        assert s.query_tel_page("C-1") == []
        s.close()


class TestCountTel:
    """W3: count_tel."""

    def test_count_zero_empty(self, store):
        assert store.count_tel("NOPE") == 0

    def test_count_all_rows(self, store):
        for i in range(5):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        assert store.count_tel("C-1") == 5

    def test_count_excludes_deleted(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        _insert_tel_row(store, "C-1", 1001.0, raw_value=2.0)
        store.soft_delete_tel("C-1", [rid])
        assert store.count_tel("C-1") == 1

    def test_count_include_deleted(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        _insert_tel_row(store, "C-1", 1001.0, raw_value=2.0)
        store.soft_delete_tel("C-1", [rid])
        assert store.count_tel("C-1", include_deleted=True) == 2

    def test_count_with_time_range(self, store):
        for i in range(10):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        assert store.count_tel("C-1", start_ts=1003.0, end_ts=1005.0) == 3

    def test_count_matches_page_total(self, store):
        """count_tel and the total reachable via pagination must agree."""
        for i in range(7):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        counted = store.count_tel("C-1")
        # Pull everything with a generous limit.
        queried = store.query_tel_page("C-1", limit=1000)
        assert counted == len(queried)

    def test_disabled_store_returns_zero(self, tmp_path):
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        assert s.count_tel("C-1") == 0
        s.close()


class TestTelSoftDeleteRestorePurge:
    """W4: soft_delete_tel / restore_tel / purge_tel / query_tel_deleted."""

    def test_soft_delete_moves_to_bin(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        n = store.soft_delete_tel("C-1", [rid])
        assert n == 1
        # Live query hides it.
        assert len(store.query_tel_page("C-1", limit=10)) == 0
        # Recycle bin shows it.
        deleted = store.query_tel_deleted("C-1", limit=10)
        assert len(deleted) == 1
        assert deleted[0]["timestamp"] == 1000.0
        assert deleted[0]["deleted_at"] is not None

    def test_soft_delete_idempotent(self, store):
        """Soft-deleting an already-deleted row is not re-counted."""
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        assert store.soft_delete_tel("C-1", [rid]) == 1
        assert store.soft_delete_tel("C-1", [rid]) == 0

    def test_soft_delete_batch(self, store):
        ids = [_insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
               for i in range(3)]
        n = store.soft_delete_tel("C-1", ids)
        assert n == 3
        assert store.count_tel("C-1") == 0

    def test_soft_delete_invalid_rowids_filtered(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        # 0 / negative / string / None are dropped by _sanitize_ids.
        n = store.soft_delete_tel("C-1", [rid, 0, -1, "abc", None])
        assert n == 1

    def test_soft_delete_empty_list(self, store):
        _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        assert store.soft_delete_tel("C-1", []) == 0
        assert store.soft_delete_tel("C-1", None) == 0

    def test_soft_delete_channel_isolation(self, store):
        """rowids are per-table — deleting in C-1 must not touch C-2."""
        rid1 = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        _insert_tel_row(store, "C-2", 1000.0, raw_value=2.0)
        store.soft_delete_tel("C-1", [rid1])
        assert store.count_tel("C-1") == 0
        assert store.count_tel("C-2") == 1

    def test_soft_delete_invalid_channel(self, store):
        assert store.soft_delete_tel("", [1]) == 0
        assert store.soft_delete_tel(None, [1]) == 0

    def test_restore_brings_row_back(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        store.soft_delete_tel("C-1", [rid])
        n = store.restore_tel("C-1", [rid])
        assert n == 1
        # Live query shows it again; bin is empty.
        assert len(store.query_tel_page("C-1", limit=10)) == 1
        assert store.query_tel_deleted("C-1", limit=10) == []
        # deleted_at cleared.
        page = store.query_tel_page("C-1", limit=10)
        assert page[0]["deleted_at"] is None

    def test_restore_only_touches_deleted(self, store):
        rid1 = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        rid2 = _insert_tel_row(store, "C-1", 1001.0, raw_value=2.0)
        store.soft_delete_tel("C-1", [rid1])
        # Pass both, but only rid1 is deleted.
        n = store.restore_tel("C-1", [rid1, rid2])
        assert n == 1

    def test_purge_permanently_removes(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        store.soft_delete_tel("C-1", [rid])
        n = store.purge_tel("C-1", [rid])
        assert n == 1
        # Physically gone from both live and bin.
        assert store.count_tel("C-1") == 0
        assert store.query_tel_deleted("C-1", limit=10) == []
        # Raw count is 0 too.
        table = _tel_table_name("C-1")
        raw = store._conn.execute(
            f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        assert raw == 0

    def test_purge_refuses_live_row(self, store):
        """Safety: purge_tel must not delete rows that are not soft-deleted."""
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        n = store.purge_tel("C-1", [rid])
        assert n == 0
        # Row still present.
        assert store.count_tel("C-1") == 1

    def test_query_tel_deleted_pagination(self, store):
        for i in range(5):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        # Soft-delete all.
        table = _tel_table_name("C-1")
        store._conn.execute(
            f'UPDATE "{table}" SET deleted_at = unixepoch() + timestamp'
        )
        store._conn.commit()
        all_del = store.query_tel_deleted("C-1", limit=50)
        assert len(all_del) == 5
        page = store.query_tel_deleted("C-1", limit=2, offset=2)
        assert len(page) == 2
        # newest-deleted-first ordering preserved.
        assert page[0]["deleted_at"] >= page[1]["deleted_at"]

    def test_disabled_store_all_noop(self, tmp_path):
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        assert s.soft_delete_tel("C-1", [1]) == 0
        assert s.restore_tel("C-1", [1]) == 0
        assert s.purge_tel("C-1", [1]) == 0
        assert s.query_tel_deleted("C-1") == []
        s.close()


class TestInsertTelManual:
    """W5: insert_tel_manual (origin='manual', INSERT OR REPLACE)."""

    def test_insert_marks_origin_manual(self, store):
        ok = store.insert_tel_manual("C-1", 1000.0, raw_value=42.0,
                                     anomaly_score=0.9)
        assert ok is True
        page = store.query_tel_page("C-1", limit=10)
        assert len(page) == 1
        assert page[0]["origin"] == "manual"
        assert page[0]["raw_value"] == 42.0
        assert page[0]["anomaly_score"] == 0.9

    def test_insert_with_predicted_value(self, store):
        ok = store.insert_tel_manual("C-1", 1000.0, predicted_value=7.5)
        assert ok is True
        page = store.query_tel_page("C-1", limit=10)
        assert page[0]["predicted_value"] == 7.5
        assert page[0]["raw_value"] is None

    def test_insert_or_replace_overwrites(self, store):
        """A second insert at the same timestamp must overwrite (UPSERT)."""
        store.insert_tel_manual("C-1", 1000.0, raw_value=1.0)
        store.insert_tel_manual("C-1", 1000.0, raw_value=99.0)
        page = store.query_tel_page("C-1", limit=10)
        assert len(page) == 1  # not 2 — same PRIMARY KEY
        assert page[0]["raw_value"] == 99.0

    def test_insert_does_not_touch_acquisition_state(self, store):
        """Manual insert must not perturb acquisition bookkeeping.
        Here we assert the telemetry count is exactly the rows we inserted
        (no phantom acquisition rows) and origin is 'manual'."""
        store.insert_tel_manual("C-1", 1000.0, raw_value=1.0)
        store.insert_tel_manual("C-1", 1001.0, raw_value=2.0)
        assert store.count_tel("C-1") == 2
        for r in store.query_tel_page("C-1", limit=10):
            assert r["origin"] == "manual"

    def test_insert_invalid_channel(self, store):
        assert store.insert_tel_manual("", 1000.0, raw_value=1.0) is False
        assert store.insert_tel_manual(None, 1000.0, raw_value=1.0) is False

    def test_insert_invalid_timestamp(self, store):
        assert store.insert_tel_manual("C-1", "not-a-number",
                                       raw_value=1.0) is False
        assert store.insert_tel_manual("C-1", None, raw_value=1.0) is False

    def test_insert_invalid_numeric_args_coerced(self, store):
        """Non-numeric raw_value / anomaly_score are coerced to None."""
        ok = store.insert_tel_manual("C-1", 1000.0, raw_value="bad",
                                     anomaly_score="alsobad")
        assert ok is True
        page = store.query_tel_page("C-1", limit=10)
        assert page[0]["raw_value"] is None
        assert page[0]["anomaly_score"] is None

    def test_disabled_store_returns_false(self, tmp_path):
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        assert s.insert_tel_manual("C-1", 1000.0, raw_value=1.0) is False
        s.close()

    def test_acquisition_rows_origin_acq(self, store):
        """Existing acquisition writes (enqueue path) must carry origin='acq'
        via the DEFAULT — proving the manual insert path doesn't interfere."""
        t0 = 2000.0
        for i in range(3):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
        time.sleep(1.0)
        page = store.query_tel_page("C-1", limit=10)
        assert len(page) == 3
        assert all(r["origin"] == "acq" for r in page)


class TestIterTelRows:
    """W6: iter_tel_rows streaming export."""

    def test_streams_all_live_rows(self, store):
        for i in range(10):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        rows = list(store.iter_tel_rows("C-1"))
        assert len(rows) == 10
        # Ascending order.
        ts = [r["timestamp"] for r in rows]
        assert ts == sorted(ts)

    def test_small_batch_size_paginates(self, store):
        """batch_size=2 must still return all rows (multi-batch keyset walk)."""
        for i in range(10):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        rows = list(store.iter_tel_rows("C-1", batch_size=2))
        assert len(rows) == 10
        ts = [r["timestamp"] for r in rows]
        # No duplicates, fully ascending.
        assert ts == sorted(ts)
        assert len(set(ts)) == 10

    def test_excludes_deleted(self, store):
        rid = _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0)
        _insert_tel_row(store, "C-1", 1001.0, raw_value=2.0)
        store.soft_delete_tel("C-1", [rid])
        rows = list(store.iter_tel_rows("C-1"))
        assert len(rows) == 1
        assert rows[0]["timestamp"] == 1001.0

    def test_time_range(self, store):
        for i in range(10):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        rows = list(store.iter_tel_rows("C-1", start_ts=1003.0, end_ts=1005.0))
        ts = [r["timestamp"] for r in rows]
        assert all(1003.0 <= t <= 1005.0 for t in ts)
        assert len(rows) == 3

    def test_empty_channel(self, store):
        assert list(store.iter_tel_rows("NOPE")) == []

    def test_row_shape(self, store):
        _insert_tel_row(store, "C-1", 1000.0, raw_value=1.0, anomaly_score=0.5)
        rows = list(store.iter_tel_rows("C-1"))
        r = rows[0]
        for key in ("id", "timestamp", "raw_value", "anomaly_score",
                    "predicted_value", "predicted_anomaly_score", "origin_ts",
                    "ingested_at", "deleted_at", "origin"):
            assert key in r

    def test_large_volume_bounded_batches(self, store):
        """5000 rows streamed with batch_size=1000 must yield exactly 5000,
        proving keyset pagination across 5 batches doesn't drop/dup rows."""
        for i in range(5000):
            _insert_tel_row(store, "C-1", 1000.0 + i, raw_value=float(i))
        rows = list(store.iter_tel_rows("C-1", batch_size=1000))
        assert len(rows) == 5000
        ts = [r["timestamp"] for r in rows]
        assert ts == sorted(ts)
        assert len(set(ts)) == 5000

    def test_disabled_store_empty(self, tmp_path):
        s = SQLiteStore(tmp_path / "noop.db", enabled=False)
        assert list(s.iter_tel_rows("C-1")) == []
        s.close()


class TestExistingInterfacesUnchanged:
    """W7: zero-breakage guarantee — existing query_window / query_history /
    delete_history must keep working after the schema + migration changes."""

    def test_query_window_still_works(self, store):
        t0 = time.time()
        for i in range(10):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i * 0.1)
        time.sleep(1.0)
        w = store.query_window("C-1", count=5)
        assert w["count"] == 5

    def test_query_history_still_works(self, store):
        t0 = time.time()
        for i in range(5):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
        time.sleep(1.0)
        rows = store.query_history("C-1")
        assert len(rows) == 5

    def test_delete_history_still_hard_deletes(self, store):
        t0 = time.time()
        for i in range(3):
            store.enqueue_telemetry("C-1", float(i), 0.1, t0 + i)
        time.sleep(1.0)
        n = store.delete_history(channel="C-1")
        assert n == 3
        assert store.query_history("C-1") == []

    def test_stats_still_works(self, store):
        store.enqueue_telemetry("C-1", 1.0, 0.5, time.time())
        time.sleep(1.0)
        stats = store.stats()
        assert stats["enabled"] is True
        assert stats["telemetry"] >= 1
