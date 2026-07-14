"""Route tests for POST /api/warnings/{id}/verdict (human annotation)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from phm.api import deps, routes_warnings
from phm.database.warning_store import WarningStore


@pytest.fixture
def client(monkeypatch):
    ws = WarningStore()
    ws.add_pending("C-1", 0.0, 1.0, 0.9)
    fake = SimpleNamespace(
        warning_service=SimpleNamespace(
            warnings=ws,
            list=lambda limit=50: ws.recent(limit),
        ),
    )
    monkeypatch.setattr(deps, "_container", fake)
    app = FastAPI()
    app.include_router(routes_warnings.router)
    yield TestClient(app)


class TestWarningVerdict:
    def test_set_human_verdict(self, client):
        warnings = client.get("/api/warnings").json()["warnings"]
        wid = warnings[0]["id"]
        r = client.post(f"/api/warnings/{wid}/verdict", json={"human_verdict": "real"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # Verify persisted in the warning
        w = client.get("/api/warnings").json()["warnings"][0]
        assert w["human_verdict"] == "real"
        assert w["final_status"] == "real"

    def test_verdict_404_unknown_id(self, client):
        r = client.post("/api/warnings/9999/verdict", json={"human_verdict": "real"})
        assert r.status_code == 404
        assert r.json()["ok"] is False

    def test_verdict_invalid_value_422(self, client):
        warnings = client.get("/api/warnings").json()["warnings"]
        wid = warnings[0]["id"]
        r = client.post(f"/api/warnings/{wid}/verdict", json={"human_verdict": "bogus"})
        assert r.status_code == 422

    def test_verdict_false_alarm(self, client):
        warnings = client.get("/api/warnings").json()["warnings"]
        wid = warnings[0]["id"]
        r = client.post(f"/api/warnings/{wid}/verdict", json={"human_verdict": "false_alarm"})
        assert r.status_code == 200
        w = client.get("/api/warnings").json()["warnings"][0]
        assert w["human_verdict"] == "false_alarm"
        assert w["final_status"] == "false_alarm"

    def test_verdict_uncertain(self, client):
        warnings = client.get("/api/warnings").json()["warnings"]
        wid = warnings[0]["id"]
        r = client.post(f"/api/warnings/{wid}/verdict", json={"human_verdict": "uncertain"})
        assert r.status_code == 200
        w = client.get("/api/warnings").json()["warnings"][0]
        assert w["final_status"] == "uncertain"
