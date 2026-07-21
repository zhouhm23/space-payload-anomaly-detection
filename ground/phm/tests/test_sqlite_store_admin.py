"""Unit tests for SQLiteStore 后台管理扩展方法（Day21 第 3/4 页共用前置）。

覆盖 ``query_deleted`` / ``delete_by_ids`` / ``restore`` / ``purge_by_ids`` /
``update_alert_verdict_by_ids`` / ``query_alerts_filtered`` /
``insert_alert_manual`` 共 7 个新方法。

设计原则（对齐 ``test_sqlite_store.py``）：
  - 用 ``tmp_path`` fixture 创建临时 db
  - 直接通过 ``store._conn.execute(...)`` 塞测试数据（绕过 enqueue 的异步 flush）
  - 每个方法 happy path + 边界（空 ids / 非法 table / 类型校验）+ 部分命中
"""

from __future__ import annotations

import os
import sys
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from phm.database.sqlite_store import SQLiteStore


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_admin.db"


@pytest.fixture
def store(db_path):
    s = SQLiteStore(db_path, batch_size=5, flush_interval=0.5, enabled=True)
    s.start()
    yield s
    s.close()


def _insert_alert(store, channel="C-1", created_at=None, status="active",
                  llm_verdict=None, human_verdict=None, score=0.7):
    """同步插一条 alert，返回新 id（不走 enqueue 的异步队列）。"""
    ts = created_at if created_at is not None else time.time()
    cur = store._conn.execute(
        "INSERT INTO alert_records "
        "(channel, alert_type, score, message, created_at, status, "
        " llm_verdict, human_verdict, ingested_at, is_deleted) "
        "VALUES (?, 'measured', ?, '', ?, ?, ?, ?, unixepoch(), 0)",
        [channel, score, ts, status, llm_verdict, human_verdict],
    )
    store._conn.commit()
    return cur.lastrowid


def _insert_detection(store, channel="C-1", timestamp=None):
    ts = timestamp if timestamp is not None else time.time()
    cur = store._conn.execute(
        "INSERT INTO detection_results "
        "(channel, timestamp, l1_score, l2_score, l3_score, final_score, "
        " ingested_at, is_deleted) VALUES (?, ?, 0.1, 0.2, 0.3, 0.4, unixepoch(), 0)",
        [channel, ts],
    )
    store._conn.commit()
    return cur.lastrowid


def _insert_diagnosis(store, channel="C-1", alert_ts=None, verdict="real"):
    ts = alert_ts if alert_ts is not None else time.time()
    cur = store._conn.execute(
        "INSERT INTO diagnosis_records "
        "(channel, alert_type, alert_ts, diagnosis, elapsed_sec, llm_verdict, "
        " created_at, is_deleted) VALUES (?, 'measured', ?, 'text', 1.0, ?, unixepoch(), 0)",
        [channel, ts, verdict],
    )
    store._conn.commit()
    return cur.lastrowid


# ── query_deleted ────────────────────────────────────────────────────────

class TestQueryDeleted:

    def test_alerts_returns_only_soft_deleted(self, store):
        a1 = _insert_alert(store, channel="C-1")
        a2 = _insert_alert(store, channel="C-2")
        # 把 a1 软删，并塞个 raw_snapshot 用来验证 raw_value 派生
        store._conn.execute(
            "UPDATE alert_records SET is_deleted=1, deleted_at=unixepoch(), "
            "raw_snapshot=? WHERE id=?",
            ['[0.10, 0.20, 0.35]', a1],
        )
        store._conn.commit()
        rows = store.query_deleted("alert_records")
        assert len(rows) == 1
        assert rows[0]["id"] == a1
        assert rows[0]["channel"] == "C-1"
        assert rows[0]["deleted_at"] is not None
        # final_status 应已派生（compute_final_status）
        assert "final_status" in rows[0]
        # raw_value 应取 raw_snapshot 末点（需求书「遥测值」列）
        assert rows[0]["raw_value"] == 0.35
        assert rows[0]["raw_snapshot"] == [0.10, 0.20, 0.35]

    def test_alerts_raw_value_none_when_no_snapshot(self, store):
        """无 raw_snapshot 时 raw_value 应为 None（不抛错）。"""
        a1 = _insert_alert(store)
        store._conn.execute(
            "UPDATE alert_records SET is_deleted=1, deleted_at=unixepoch() WHERE id=?",
            [a1],
        )
        store._conn.commit()
        rows = store.query_deleted("alert_records")
        assert len(rows) == 1
        assert rows[0]["raw_value"] is None
        assert rows[0]["raw_snapshot"] is None

    def test_alerts_raw_value_none_when_snapshot_garbage(self, store):
        """raw_snapshot 是非法 JSON 时，raw_value=None 不抛错。"""
        a1 = _insert_alert(store)
        store._conn.execute(
            "UPDATE alert_records SET is_deleted=1, deleted_at=unixepoch(), "
            "raw_snapshot=? WHERE id=?",
            ['not-json', a1],
        )
        store._conn.commit()
        rows = store.query_deleted("alert_records")
        assert len(rows) == 1
        assert rows[0]["raw_value"] is None

    def test_detections_and_diagnoses(self, store):
        d1 = _insert_detection(store)
        diag1 = _insert_diagnosis(store)
        store._conn.execute(
            "UPDATE detection_results SET is_deleted=1, deleted_at=unixepoch() WHERE id=?",
            [d1],
        )
        store._conn.execute(
            "UPDATE diagnosis_records SET is_deleted=1, deleted_at=unixepoch() WHERE id=?",
            [diag1],
        )
        store._conn.commit()
        d_rows = store.query_deleted("detection_results")
        assert len(d_rows) == 1 and d_rows[0]["id"] == d1
        assert "final_score" in d_rows[0]
        diag_rows = store.query_deleted("diagnosis_records")
        assert len(diag_rows) == 1 and diag_rows[0]["id"] == diag1
        assert diag_rows[0]["llm_verdict"] == "real"

    def test_invalid_table_returns_empty(self, store):
        _insert_alert(store)
        assert store.query_deleted("telemetry_C-1") == []
        assert store.query_deleted("unknown_table") == []
        assert store.query_deleted("") == []

    def test_disabled_store_returns_empty(self, db_path):
        s = SQLiteStore(db_path, enabled=False)
        s.start()
        try:
            assert s.query_deleted("alert_records") == []
        finally:
            s.close()

    def test_limit_cap(self, store):
        for _ in range(5):
            _insert_alert(store)
        store._conn.execute(
            "UPDATE alert_records SET is_deleted=1, deleted_at=unixepoch()"
        )
        store._conn.commit()
        rows = store.query_deleted("alert_records", limit=3)
        assert len(rows) == 3
        # 非法 limit 兜底
        rows2 = store.query_deleted("alert_records", limit="abc")
        assert len(rows2) <= 200  # fallback


# ── delete_by_ids ────────────────────────────────────────────────────────

class TestDeleteByIds:

    def test_soft_delete_matching(self, store):
        a1 = _insert_alert(store)
        a2 = _insert_alert(store)
        a3 = _insert_alert(store)
        n = store.delete_by_ids("alert_records", [a1, a3])
        assert n == 2
        # a2 应仍可查
        rows = store.query_alerts(limit=10)
        ids = [r["id"] for r in rows]
        assert a2 in ids and a1 not in ids and a3 not in ids

    def test_empty_ids_short_circuits(self, store):
        _insert_alert(store)
        assert store.delete_by_ids("alert_records", []) == 0
        assert store.delete_by_ids("alert_records", None) == 0

    def test_invalid_table(self, store):
        a1 = _insert_alert(store)
        assert store.delete_by_ids("unknown", [a1]) == 0

    def test_already_deleted_not_recounted(self, store):
        a1 = _insert_alert(store)
        store.delete_by_ids("alert_records", [a1])
        # 再删一次，不应计数
        assert store.delete_by_ids("alert_records", [a1]) == 0

    def test_invalid_ids_filtered(self, store):
        a1 = _insert_alert(store)
        # 含 0/负数/字符串，应被 _sanitize_ids 过滤掉，只保留 a1
        n = store.delete_by_ids("alert_records", [a1, 0, -5, "abc", None])
        assert n == 1


# ── restore ──────────────────────────────────────────────────────────────

class TestRestore:

    def test_restore_makes_row_visible(self, store):
        a1 = _insert_alert(store)
        store.delete_by_ids("alert_records", [a1])
        n = store.restore("alert_records", [a1])
        assert n == 1
        # 现在应能查到
        rows = store.query_alerts(limit=10)
        assert any(r["id"] == a1 for r in rows)
        # deleted_at 应被清空
        row = store._conn.execute(
            "SELECT is_deleted, deleted_at FROM alert_records WHERE id=?", [a1]
        ).fetchone()
        assert row[0] == 0 and row[1] is None

    def test_restore_only_touches_deleted(self, store):
        a1 = _insert_alert(store)
        a2 = _insert_alert(store)
        store.delete_by_ids("alert_records", [a1])
        # 传 a1 + a2，但只有 a1 是软删态
        n = store.restore("alert_records", [a1, a2])
        assert n == 1

    def test_empty_ids(self, store):
        _insert_alert(store)
        assert store.restore("alert_records", []) == 0

    def test_invalid_table(self, store):
        a1 = _insert_alert(store)
        store.delete_by_ids("alert_records", [a1])
        assert store.restore("unknown", [a1]) == 0


# ── purge_by_ids ─────────────────────────────────────────────────────────

class TestPurgeByIds:

    def test_purge_physically_removes(self, store):
        a1 = _insert_alert(store)
        store.delete_by_ids("alert_records", [a1])
        n = store.purge_by_ids("alert_records", [a1])
        assert n == 1
        # 行应已物理消失
        row = store._conn.execute(
            "SELECT COUNT(*) FROM alert_records WHERE id=?", [a1]
        ).fetchone()
        assert row[0] == 0

    def test_purge_refuses_active_row(self, store):
        """关键安全：purge_by_ids 不能删 is_deleted=0 的行。"""
        a1 = _insert_alert(store)
        n = store.purge_by_ids("alert_records", [a1])
        assert n == 0
        # 行还在
        rows = store.query_alerts(limit=10)
        assert any(r["id"] == a1 for r in rows)

    def test_empty_ids(self, store):
        _insert_alert(store)
        assert store.purge_by_ids("alert_records", []) == 0

    def test_invalid_table(self, store):
        a1 = _insert_alert(store)
        store.delete_by_ids("alert_records", [a1])
        assert store.purge_by_ids("unknown", [a1]) == 0


# ── update_alert_verdict_by_ids ──────────────────────────────────────────

class TestUpdateAlertVerdictByIds:

    def test_batch_set_human_verdict(self, store):
        a1 = _insert_alert(store)
        a2 = _insert_alert(store)
        a3 = _insert_alert(store)
        n = store.update_alert_verdict_by_ids([a1, a2, a3], "real", is_llm=False)
        assert n == 3
        rows = store.query_alerts(limit=10)
        for r in rows:
            if r["id"] in (a1, a2, a3):
                assert r["human_verdict"] == "real"

    def test_is_llm_writes_llm_column(self, store):
        a1 = _insert_alert(store)
        store.update_alert_verdict_by_ids([a1], "false_alarm", is_llm=True)
        rows = store.query_alerts(limit=10)
        r = next(x for x in rows if x["id"] == a1)
        assert r["llm_verdict"] == "false_alarm"
        assert r["human_verdict"] is None

    def test_invalid_verdict_rejected(self, store):
        a1 = _insert_alert(store)
        assert store.update_alert_verdict_by_ids([a1], "bogus") == 0
        assert store.update_alert_verdict_by_ids([a1], "") == 0

    def test_skips_deleted_rows(self, store):
        a1 = _insert_alert(store)
        store.delete_by_ids("alert_records", [a1])
        n = store.update_alert_verdict_by_ids([a1], "real")
        assert n == 0


# ── query_alerts_filtered ────────────────────────────────────────────────

class TestQueryAlertsFiltered:

    def test_filter_by_channel(self, store):
        _insert_alert(store, channel="C-1")
        _insert_alert(store, channel="C-2")
        _insert_alert(store, channel="C-1")
        rows = store.query_alerts_filtered(channel="C-1", limit=50)
        assert len(rows) == 2
        assert all(r["channel"] == "C-1" for r in rows)

    def test_filter_by_alert_type(self, store):
        _insert_alert(store)  # measured
        # 手动塞一条 predicted
        store._conn.execute(
            "INSERT INTO alert_records (channel, alert_type, created_at, ingested_at, is_deleted) "
            "VALUES ('C-1', 'predicted', ?, unixepoch(), 0)",
            [time.time()],
        )
        store._conn.commit()
        rows = store.query_alerts_filtered(alert_type="measured", limit=50)
        assert all(r["alert_type"] == "measured" for r in rows)
        rows_p = store.query_alerts_filtered(alert_type="predicted", limit=50)
        assert all(r["alert_type"] == "predicted" for r in rows_p)

    def test_filter_by_status(self, store):
        _insert_alert(store, status="active")
        _insert_alert(store, status="confirmed")
        rows = store.query_alerts_filtered(status="confirmed", limit=50)
        assert len(rows) == 1 and rows[0]["status"] == "confirmed"

    def test_filter_by_verdict_matches_llm_or_human(self, store):
        _insert_alert(store, llm_verdict="real")
        _insert_alert(store, human_verdict="real")
        _insert_alert(store, llm_verdict="false_alarm")
        rows = store.query_alerts_filtered(verdict="real", limit=50)
        assert len(rows) == 2
        for r in rows:
            assert r["llm_verdict"] == "real" or r["human_verdict"] == "real"

    def test_filter_by_time_range(self, store):
        t0 = time.time()
        _insert_alert(store, created_at=t0 - 100)
        _insert_alert(store, created_at=t0 - 50)
        _insert_alert(store, created_at=t0)
        rows = store.query_alerts_filtered(start_ts=t0 - 60, end_ts=t0 - 40, limit=50)
        assert len(rows) == 1

    def test_limit_cap_and_order(self, store):
        for i in range(5):
            _insert_alert(store, created_at=time.time() - i)
        rows = store.query_alerts_filtered(limit=3)
        assert len(rows) == 3
        # 返回按时间升序（query_alerts 同款 reversed）
        ts = [r["created_at"] for r in rows]
        assert ts == sorted(ts)

    def test_excludes_deleted(self, store):
        a1 = _insert_alert(store)
        _insert_alert(store)
        store.delete_by_ids("alert_records", [a1])
        rows = store.query_alerts_filtered(limit=50)
        assert all(r["id"] != a1 for r in rows)

    def test_snapshot_json_parsed(self, store):
        """raw_snapshot / score_snapshot 应被解析成 list（与 query_alerts 一致）。"""
        store._conn.execute(
            "INSERT INTO alert_records (channel, alert_type, created_at, status, "
            "raw_snapshot, score_snapshot, ingested_at, is_deleted) "
            "VALUES ('C-1', 'measured', ?, 'active', '[1,2,3]', '[0.1,0.2]', unixepoch(), 0)",
            [time.time()],
        )
        store._conn.commit()
        rows = store.query_alerts_filtered(limit=10)
        assert len(rows) == 1
        assert rows[0]["raw_snapshot"] == [1, 2, 3]
        assert rows[0]["score_snapshot"] == [0.1, 0.2]


# ── insert_alert_manual ──────────────────────────────────────────────────

class TestInsertAlertManual:

    def test_insert_returns_id(self, store):
        new_id = store.insert_alert_manual("C-1", score=0.9, message="manual")
        assert new_id is not None and new_id > 0
        rows = store.query_alerts(limit=10)
        r = next(x for x in rows if x["id"] == new_id)
        assert r["channel"] == "C-1"
        assert r["alert_type"] == "measured"
        assert r["score"] == 0.9
        assert r["message"] == "manual"
        assert r["status"] == "active"

    def test_invalid_channel_returns_none(self, store):
        assert store.insert_alert_manual("", score=0.9) is None
        assert store.insert_alert_manual(None, score=0.9) is None

    def test_invalid_score_handled(self, store):
        """非数值 score 应被静默转 None（不抛错）。"""
        new_id = store.insert_alert_manual("C-1", score="not-a-number")
        assert new_id is not None
        rows = store.query_alerts(limit=10)
        r = next(x for x in rows if x["id"] == new_id)
        assert r["score"] is None

    def test_created_at_persisted(self, store):
        ts = time.time() - 3600
        new_id = store.insert_alert_manual("C-1", score=0.5, created_at=ts)
        rows = store.query_alerts(limit=10)
        r = next(x for x in rows if x["id"] == new_id)
        assert abs(r["created_at"] - ts) < 0.001

    def test_snapshots_persisted(self, store):
        new_id = store.insert_alert_manual(
            "C-1", score=0.5,
            raw_snapshot=[1.0, 2.0, 3.0],
            score_snapshot=[0.1, 0.2, 0.3],
        )
        rows = store.query_alerts(limit=10)
        r = next(x for x in rows if x["id"] == new_id)
        assert r["raw_snapshot"] == [1.0, 2.0, 3.0]
        assert r["score_snapshot"] == [0.1, 0.2, 0.3]


# ── 纯函数 _sanitize_ids / _placeholders ─────────────────────────────────

class TestSanitizeIds:

    def test_dedup_and_filter(self, store):
        out = store._sanitize_ids([1, 2, 2, 3, 0, -1, "abc", None, "4"])
        assert out == [1, 2, 3, 4]

    def test_empty_input(self, store):
        assert store._sanitize_ids([]) == []
        assert store._sanitize_ids(None) == []
        assert store._sanitize_ids([0, -1, None, "x"]) == []

    def test_placeholders(self, store):
        assert store._placeholders(1) == "?"
        assert store._placeholders(3) == "?,?,?"
        assert store._placeholders(0) == "?"  # max(n,1)
