"""Route tests for GET /api/alerts/history and PATCH /api/alerts/{id}.

``GET /api/alerts/history`` reads the SQLite ``alert_records`` table (with
``id`` + ``status``), distinct from ``GET /api/alerts`` which reads the
in-memory deque.  ``PATCH /api/alerts/{id}`` updates the lifecycle status.
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

from phm.api import deps, alerts_router
from phm.database.sqlite_store import SQLiteStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with only the alerts router and a temp SQLite store."""
    sqlite = SQLiteStore(
        tmp_path / "t.db", batch_size=5, flush_interval=0.5, enabled=True,
    )
    sqlite.start()
    # routes_alerts uses c.sqlite and c.alert_service.threshold
    fake = SimpleNamespace(
        sqlite=sqlite,
        alert_service=SimpleNamespace(threshold=0.5),
    )
    monkeypatch.setattr(deps, "_container", fake)
    app = FastAPI()
    app.include_router(alerts_router)
    yield TestClient(app)
    sqlite.close()


def _seed_alert(store, status="pending", channel="C-1", score=0.85):
    store.enqueue_alert({
        "channel": channel, "type": "predicted", "score": score,
        "message": "seed", "time": time.time(), "status": status,
    })
    time.sleep(1.0)
    return store.query_alerts()[0]["id"]


class TestAlertsHistory:

    def test_history_returns_id_and_status(self, client):
        store = deps.get().sqlite
        _seed_alert(store, status="pending")
        resp = client.get("/api/alerts/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["threshold"] == 0.5
        assert len(body["alerts"]) == 1
        a = body["alerts"][0]
        assert a["id"] == 1
        assert a["status"] == "pending"
        assert a["channel"] == "C-1"

    def test_history_limit(self, client):
        store = deps.get().sqlite
        for i in range(5):
            store.enqueue_alert({
                "channel": "C-1", "type": "measured", "score": 0.8,
                "message": f"a{i}", "time": time.time() + i,
            })
        time.sleep(1.0)
        resp = client.get("/api/alerts/history", params={"limit": 2})
        assert resp.status_code == 200
        assert len(resp.json()["alerts"]) == 2


class TestPatchAlert:

    def test_patch_confirmed(self, client):
        store = deps.get().sqlite
        aid = _seed_alert(store, status="pending")
        resp = client.patch(f"/api/alerts/{aid}", json={"status": "confirmed"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["id"] == aid
        assert body["status"] == "confirmed"
        # Persisted
        assert store.query_alerts()[0]["status"] == "confirmed"
        assert store.query_alerts()[0]["verified_at"] is not None

    def test_patch_false(self, client):
        store = deps.get().sqlite
        aid = _seed_alert(store, status="pending")
        resp = client.patch(f"/api/alerts/{aid}", json={"status": "false"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "false"
        assert store.query_alerts()[0]["status"] == "false"

    def test_patch_missing_id_404(self, client):
        resp = client.patch("/api/alerts/99999", json={"status": "confirmed"})
        assert resp.status_code == 404
        assert resp.json()["ok"] is False

    def test_patch_invalid_status_404(self, client):
        """``active`` is not patchable → treated as failure → 404."""
        store = deps.get().sqlite
        aid = _seed_alert(store, status="pending")
        resp = client.patch(f"/api/alerts/{aid}", json={"status": "active"})
        assert resp.status_code == 404
        # Unchanged
        assert store.query_alerts()[0]["status"] == "pending"

    def test_patch_bogus_status_404(self, client):
        store = deps.get().sqlite
        aid = _seed_alert(store, status="pending")
        resp = client.patch(f"/api/alerts/{aid}", json={"status": "bogus"})
        assert resp.status_code == 404
        assert store.query_alerts()[0]["status"] == "pending"


class TestAlertVerdict:
    """Tests for POST /api/alerts/verdict (human annotation on measured alerts)."""

    def test_set_alert_verdict(self, client):
        store = deps.get().sqlite
        _seed_alert(store, channel="C-1")
        # Use a known timestamp — _seed_alert uses time.time(), so we
        # query to get the actual created_at.
        alert = store.query_alerts()[0]
        resp = client.post("/api/alerts/verdict", json={
            "channel": "C-1", "alert_ts": alert["created_at"],
            "human_verdict": "real",
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Verify persisted
        assert store.query_alerts()[0]["human_verdict"] == "real"

    def test_alert_verdict_404_not_found(self, client):
        resp = client.post("/api/alerts/verdict", json={
            "channel": "ZZZ", "alert_ts": 99999.0,
            "human_verdict": "real",
        })
        assert resp.status_code == 404

    def test_alert_verdict_invalid_value(self, client):
        store = deps.get().sqlite
        _seed_alert(store, channel="C-1")
        alert = store.query_alerts()[0]
        resp = client.post("/api/alerts/verdict", json={
            "channel": "C-1", "alert_ts": alert["created_at"],
            "human_verdict": "bogus",
        })
        assert resp.status_code == 422
