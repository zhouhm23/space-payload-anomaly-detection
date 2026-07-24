"""Functional test: verify recycle filtering at the SQLiteStore level."""
import os
import sys
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # noqa
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from phm.database.sqlite_store import SQLiteStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "test_recycle.db", batch_size=5, flush_interval=0.5, enabled=True)
    s.start()
    yield s
    s.close()


def test_alert_recycle_deletion_time_filter(store):
    for i in range(3):
        rid = store.insert_alert_manual('C-1', 0.5 * i, 'm%d' % i, created_at=1700000000.0 + i)
        store.delete_by_ids('alert_records', [rid])
    with store._write_lock:
        store._conn.execute('UPDATE alert_records SET deleted_at=? WHERE id=1', [1700001000.0])
        store._conn.execute('UPDATE alert_records SET deleted_at=? WHERE id=2', [1700002000.0])
        store._conn.execute('UPDATE alert_records SET deleted_at=? WHERE id=3', [1700003000.0])
        store._conn.commit()

    assert store.count_deleted('alert_records') == 3
    assert store.count_deleted('alert_records', deleted_start=1700001500.0) == 2
    assert store.count_deleted('alert_records', deleted_end=1700001500.0) == 1
    rows = store.query_deleted('alert_records', deleted_start=1700001500.0)
    assert sorted(r['id'] for r in rows) == [2, 3]


def test_alert_recycle_channel_filter(store):
    for ch in ('C-1', 'C-2'):
        rid = store.insert_alert_manual(ch, 0.5, 'm', created_at=1700000000.0)
        store.delete_by_ids('alert_records', [rid])
    with store._write_lock:
        store._conn.execute('UPDATE alert_records SET deleted_at=?', [1700001000.0])
        store._conn.commit()
    assert store.count_deleted('alert_records', channel='C-1') == 1
    assert store.count_deleted('alert_records', channel='NOPE') == 0
    rows = store.query_deleted('alert_records', channel='C-2')
    assert len(rows) == 1 and rows[0]['channel'] == 'C-2'


def test_alert_recycle_status_filter(store):
    rid = store.insert_alert_manual('C-1', 0.5, 'm', created_at=1700000000.0)
    store.delete_by_ids('alert_records', [rid])
    with store._write_lock:
        store._conn.execute('UPDATE alert_records SET deleted_at=?, llm_verdict=? WHERE id=1',
                            [1700001000.0, 'real'])
        store._conn.commit()
    rows = store.query_deleted('alert_records', status='real')
    assert sorted(r['id'] for r in rows) == [1]
    assert store.query_deleted('alert_records', status='false_alarm') == []


def test_status_filter_ignored_on_detections(store):
    store._conn.execute(
        'INSERT INTO detection_results(channel, timestamp, l1_score, is_deleted, deleted_at) '
        'VALUES(?, ?, ?, 1, ?)', ('C-1', 1.0, 0.1, 1700001000.0))
    store._conn.commit()
    # status should be silently ignored (detection_results has no status concept)
    rows = store.query_deleted('detection_results', status='real')
    assert len(rows) == 1
    # backward-compat call (no kwargs)
    assert store.count_deleted('detection_results') == 1
    assert len(store.query_deleted('detection_results', 200, 0)) == 1


def test_telemetry_recycle_deletion_time_filter(store):
    # Create a telemetry table and insert soft-deleted rows at distinct times.
    # NB: channel 'C-1' → table name 'telemetry_C_1' (non-alnum → '_').
    table = store._ensure_tel_table('C-1')
    with store._write_lock:
        store._conn.executemany(
            f'INSERT INTO "{table}"(timestamp, raw_value, deleted_at) VALUES (?, ?, ?)',
            [(1.0, 0.1, 1700001000.0), (2.0, 0.2, 1700002000.0), (3.0, 0.3, 1700003000.0)])
        store._conn.commit()
    # No filter → 3
    assert store.count_tel_deleted('C-1') == 3
    # deleted_start excludes the first
    assert store.count_tel_deleted('C-1', deleted_start=1700001500.0) == 2
    # deleted_end excludes the last two
    assert store.count_tel_deleted('C-1', deleted_end=1700001500.0) == 1
    rows = store.query_tel_deleted('C-1', deleted_start=1700001500.0)
    assert sorted(r['timestamp'] for r in rows) == [2.0, 3.0]
    # backward compat: query_tel_deleted with no date kwargs still works
    assert len(store.query_tel_deleted('C-1')) == 3
