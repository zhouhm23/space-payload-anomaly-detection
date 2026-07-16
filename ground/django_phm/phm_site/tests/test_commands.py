"""Tests for agent-friendly management commands.

Uses Django's ``call_command`` to invoke each CLI command and captures
stdout via ``StringIO``.  Service-layer methods are mocked where they
would have side effects (file writes, LLM calls, SQLite mutations) so the
tests stay hermetic.

Covers the four new commands added in Day18-续2:
  * device  (show / save / rm)
  * alert   (list / verdict / status)
  * diagnose (run / auto / status / list)
  * export  (telemetry — csv / json / --last / -o)

The original three commands (rul / models / config) are exercised
end-to-end by the existing HTTP-view tests via the shared service layer,
so they are not re-tested here.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from django.core.management import call_command
from django.test import TestCase

from phm_site import services_bridge


def _run(command: str, *args, **opts) -> tuple[str, str]:
    """Invoke a management command, returning (stdout, stderr) strings."""
    out = io.StringIO()
    err = io.StringIO()
    call_command(command, *args, stdout=out, stderr=err, **opts)
    return out.getvalue(), err.getvalue()


def _run_json(command: str, *args, **opts) -> dict:
    """Invoke a command with ``--format json`` and parse the JSON stdout."""
    out, _err = _run(command, *args, format="json", **opts)
    return json.loads(out)


# ── device ────────────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestDeviceCommand(TestCase):
    def setUp(self):
        services_bridge.start()

    def test_show_text_has_summary(self):
        out, _ = _run("device", "show")
        assert "Device tree" in out
        assert "Folders:" in out
        assert "Sensors:" in out

    def test_show_json_status_ok(self):
        data = _run_json("device", "show")
        assert data["status"] == "ok"
        assert "tree" in data
        assert "folders" in data and "sensors" in data

    def test_save_without_confirm_refused(self):
        data = _run_json("device", "save", "/tmp/nonexistent.json")
        assert data["status"] == "error"
        assert "--confirm" in data["message"]

    def test_save_nonexistent_file(self):
        data = _run_json("device", "save", "/tmp/does_not_exist.json", confirm=True)
        assert data["status"] == "error"
        assert "not found" in data["message"].lower()

    @patch("phm.services.config_service.ConfigService.save")
    def test_save_calls_service_save(self, mock_save):
        mock_save.return_value = {"status": "ok"}
        # Use a real temp file with a minimal tree so the command can read it.
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"device_tree": [
                {"id": "x", "type": "sensor", "sourceId": "virtual:sine",
                 "channelName": "VS-sine", "blockSize": 512}
            ]}, f)
            path = f.name
        try:
            data = _run_json("device", "save", path, confirm=True)
            assert data["status"] == "ok"
            assert mock_save.called
        finally:
            Path(path).unlink(missing_ok=True)

    @patch("phm.services.config_service.ConfigService.save")
    @patch("phm.services.config_service.ConfigService.load")
    def test_rm_removes_node(self, mock_load, mock_save):
        mock_load.return_value = {
            "device_tree": [
                {"id": "F1", "type": "folder", "children": [
                    {"id": "S1", "type": "sensor", "sourceId": "virtual:sine",
                     "channelName": "VS-sine", "blockSize": 512},
                ]},
            ],
            "aggregation_strategy": "min",
        }
        mock_save.return_value = {"status": "ok"}
        data = _run_json("device", "rm", "S1", confirm=True)
        assert data["status"] == "ok"
        assert data["removed"] == "S1"

    @patch("phm.services.config_service.ConfigService.save")
    @patch("phm.services.config_service.ConfigService.load")
    def test_rm_nonexistent_returns_not_found(self, mock_load, mock_save):
        mock_load.return_value = {"device_tree": [], "aggregation_strategy": "min"}
        data = _run_json("device", "rm", "nope", confirm=True)
        assert data["status"] == "not_found"
        assert data["node_id"] == "nope"

    def test_rm_without_confirm_refused(self):
        data = _run_json("device", "rm", "S1")
        assert data["status"] == "error"
        assert "--confirm" in data["message"]


# ── alert ─────────────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestAlertCommand(TestCase):
    def setUp(self):
        services_bridge.start()

    def test_list_json_status_ok(self):
        data = _run_json("alert", "list", limit=5)
        assert data["status"] == "ok"
        assert "alerts" in data and "threshold" in data

    def test_list_live_uses_alert_service(self):
        data = _run_json("alert", "list", limit=3, live=True)
        assert data["status"] == "ok"
        assert "live" in data["source"]

    @patch("phm.database.sqlite_store.SQLiteStore.update_alert_verdict")
    def test_verdict_success(self, mock_update):
        mock_update.return_value = True
        data = _run_json("alert", "verdict", "C-1", "123.0", "real")
        assert data["status"] == "ok"
        assert data["human_verdict"] == "real"
        assert data["channel"] == "C-1"

    @patch("phm.database.sqlite_store.SQLiteStore.update_alert_verdict")
    def test_verdict_not_found(self, mock_update):
        mock_update.return_value = False
        data = _run_json("alert", "verdict", "C-1", "123.0", "real")
        assert data["status"] == "not_found"

    def test_verdict_invalid_value(self):
        data = _run_json("alert", "verdict", "C-1", "123.0", "bogus")
        assert data["status"] == "error"
        assert "real" in data["message"]  # lists valid values

    def test_verdict_bad_ts(self):
        data = _run_json("alert", "verdict", "C-1", "abc", "real")
        assert data["status"] == "error"
        assert "number" in data["message"].lower()

    def test_verdict_wrong_arity(self):
        data = _run_json("alert", "verdict", "C-1")
        assert data["status"] == "error"

    @patch("phm.database.sqlite_store.SQLiteStore.update_alert_status")
    def test_status_success(self, mock_update):
        mock_update.return_value = True
        data = _run_json("alert", "status", "12", "confirmed")
        assert data["status"] == "ok"
        assert data["new_status"] == "confirmed"
        assert data["id"] == 12

    @patch("phm.database.sqlite_store.SQLiteStore.update_alert_status")
    def test_status_not_found(self, mock_update):
        mock_update.return_value = False
        data = _run_json("alert", "status", "999", "pending")
        assert data["status"] == "not_found"

    def test_status_invalid_value(self):
        data = _run_json("alert", "status", "12", "bogus")
        assert data["status"] == "error"


# ── diagnose ──────────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestDiagnoseCommand(TestCase):
    def setUp(self):
        services_bridge.start()

    def test_run_without_channel(self):
        data = _run_json("diagnose", "run")
        assert data["status"] == "error"
        assert "channel" in data["message"].lower()

    def test_run_bad_ts(self):
        data = _run_json("diagnose", "run", "C-1", ts="abc")
        assert data["status"] == "error"
        assert "ts" in data["message"].lower()

    @patch("phm.services.diagnosis_service.DiagnosisService.diagnose")
    def test_run_success(self, mock_diag):
        mock_diag.return_value = {
            "channel": "C-1", "alert_type": "measured",
            "diagnosis": "## Verdict\nVERDICT: real",
            "context_summary": {}, "elapsed_sec": 1.2,
            "error": None, "llm_verdict": "real", "cached": False,
        }
        data = _run_json("diagnose", "run", "C-1", type="measured", ts="123.0")
        assert data["status"] == "ok"
        assert data["llm_verdict"] == "real"
        assert data["cached"] is False

    @patch("phm.services.diagnosis_service.DiagnosisService.diagnose")
    def test_run_error_path(self, mock_diag):
        mock_diag.return_value = {
            "channel": "C-1", "alert_type": "measured",
            "diagnosis": "", "context_summary": {}, "elapsed_sec": 0.1,
            "error": "no detection data available for channel C-1",
            "llm_verdict": None, "cached": False,
        }
        data = _run_json("diagnose", "run", "C-1")
        assert data["status"] == "error"
        assert "no detection data" in data["message"]

    @patch("phm.services.diagnosis_service.DiagnosisService.auto_diagnose_all")
    def test_auto_started(self, mock_auto):
        mock_auto.return_value = {"started": True, "total": 5}
        data = _run_json("diagnose", "auto")
        assert data["status"] == "ok"
        assert data["total"] == 5

    @patch("phm.services.diagnosis_service.DiagnosisService.auto_diagnose_all")
    def test_auto_busy(self, mock_auto):
        mock_auto.return_value = {"started": False, "error": "already running"}
        data = _run_json("diagnose", "auto")
        assert data["status"] == "error"

    def test_status_json(self):
        data = _run_json("diagnose", "status")
        assert data["status"] == "ok"
        assert "running" in data and "done" in data and "total" in data

    def test_list_json(self):
        data = _run_json("diagnose", "list", limit=10)
        assert data["status"] == "ok"
        assert "diagnoses" in data


# ── export ────────────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestExportCommand(TestCase):
    def setUp(self):
        services_bridge.start()

    @patch("phm.database.sqlite_store.SQLiteStore.query_history")
    def test_csv_has_header(self, mock_q):
        mock_q.return_value = [
            {"channel": "C-1", "received_at": 1752633600.0, "raw": 0.5, "score": 0.1},
        ]
        out, _ = _run("export", "telemetry", channels="C-1",
                      start=1752633600.0, end=1752633700.0)
        lines = out.strip().splitlines()
        assert lines[0] == "channel,timestamp,raw_value,anomaly_score,received_at_iso"
        assert "C-1" in lines[1]

    @patch("phm.database.sqlite_store.SQLiteStore.query_history")
    def test_json_output(self, mock_q):
        mock_q.return_value = [
            {"channel": "C-1", "received_at": 1752633600.0, "raw": 0.5, "score": 0.1},
        ]
        out, _ = _run("export", "telemetry", channels="C-1",
                      start=1752633600.0, end=1752633700.0, format="json")
        data = json.loads(out)
        assert isinstance(data, list)
        assert data[0]["channel"] == "C-1"
        assert data[0]["raw_value"] == 0.5

    @patch("phm.database.sqlite_store.SQLiteStore.query_history")
    def test_last_window(self, mock_q):
        mock_q.return_value = []
        _run("export", "telemetry", channels="C-1", last="1h")
        # Verify the service was called with a start ≈ now - 3600.
        args, kwargs = mock_q.call_args
        start = kwargs.get("start_time")
        import time as _time
        assert start is not None
        assert abs(start - (_time.time() - 3600)) < 5  # within 5s slack

    @patch("phm.database.sqlite_store.SQLiteStore.query_history")
    def test_write_to_file(self, mock_q):
        mock_q.return_value = [
            {"channel": "C-1", "received_at": 1752633600.0, "raw": 0.5, "score": 0.1},
        ]
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            out, _ = _run("export", "telemetry", channels="C-1",
                          start=1752633600.0, end=1752633700.0, output=path)
            assert "Wrote 1 rows" in out
            written = Path(path).read_text(encoding="utf-8-sig")
            assert "channel,timestamp" in written
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_last_value(self):
        out, err = _run("export", "telemetry", channels="C-1", last="xyz")
        # Error message goes to stderr; nothing emitted on stdout.
        assert "Invalid --last" in err
