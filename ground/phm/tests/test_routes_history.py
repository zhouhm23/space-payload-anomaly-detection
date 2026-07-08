"""Route tests for DELETE /api/history and DELETE /api/detection.

These cover the ``confirm`` guard (refuse to clear the whole table without
``confirm=true``) and the happy-path deletion.  The store-level behaviour
is already covered by ``test_sqlite_store.TestStoreMutations``; here we
only verify the HTTP wiring and the guard.
"""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from phm.api import deps, history_router
from phm.database.sqlite_store import SQLiteStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with only the history router and a temp SQLite store.

    ``deps._container`` is replaced with a minimal namespace exposing just
    ``sqlite`` and ``warning_service`` (the latter only used by
    ``GET /api/detection``; not exercised by the DELETE tests).
    """
    sqlite = SQLiteStore(
        tmp_path / "t.db", batch_size=5, flush_interval=0.5, enabled=True,
    )
    sqlite.start()
    fake = SimpleNamespace(
        sqlite=sqlite,
        warning_service=SimpleNamespace(get_latest_cascade=lambda ch: None),
    )
    monkeypatch.setattr(deps, "_container", fake)
    app = FastAPI()
    app.include_router(history_router)
    yield TestClient(app)
    sqlite.close()


def _seed_telemetry(store, channel="C-1", n=5):
    t0 = time.time()
    for i in range(n):
        store.enqueue_telemetry(channel, float(i), 0.1, t0 + i)
    time.sleep(1.0)
    return t0


class TestDeleteHistory:

    def test_delete_by_channel(self, client):
        store = deps.get().sqlite
        _seed_telemetry(store, "C-1", 5)
        _seed_telemetry(store, "C-2", 3)
        resp = client.delete("/api/history", params={"channel": "C-1"})
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 5
        assert store.query_history("C-1") == []
        assert len(store.query_history("C-2")) == 3

    def test_delete_by_time_range(self, client):
        store = deps.get().sqlite
        t0 = _seed_telemetry(store, "C-1", 10)
        resp = client.delete("/api/history", params={
            "channel": "C-1", "start": t0 + 3, "end": t0 + 6,
        })
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 4

    def test_delete_all_without_confirm_refused(self, client):
        """No filter + no confirm → 400, nothing deleted."""
        store = deps.get().sqlite
        _seed_telemetry(store, "C-1", 3)
        resp = client.delete("/api/history")
        assert resp.status_code == 400
        assert resp.json()["error"] == "confirm_required"
        assert len(store.query_history()) == 3  # untouched

    def test_delete_all_with_confirm(self, client):
        store = deps.get().sqlite
        _seed_telemetry(store, "C-1", 3)
        resp = client.delete("/api/history", params={"confirm": True})
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 3
        assert store.query_history() == []


class TestDeleteDetection:

    def test_delete_by_channel(self, client):
        store = deps.get().sqlite
        # Reuse telemetry rows as a cheap way to have something to delete is
        # not correct — detection_results is a separate table.  Insert via
        # the public enqueue_detection API using a minimal cascade.
        from phm.algorithm.cascade_types import (
            LayerResult, CascadeOutput, LAYER_L1_CLASSIC, DECISION_PASS,
        )
        import numpy as np
        cascade = CascadeOutput(
            channel="C-1",
            final_scores=np.array([0.1], dtype=np.float32),
            layers=[LayerResult(LAYER_L1_CLASSIC, DECISION_PASS, 0.0, {})],
        )
        store.enqueue_detection("C-1", time.time(), cascade)
        store.enqueue_detection("C-2", time.time(), cascade)
        time.sleep(1.0)
        resp = client.delete("/api/detection", params={"channel": "C-1"})
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1
        assert store.query_detection("C-1") == []
        assert len(store.query_detection("C-2")) == 1

    def test_delete_all_without_confirm_refused(self, client):
        resp = client.delete("/api/detection")
        assert resp.status_code == 400
        assert resp.json()["error"] == "confirm_required"
