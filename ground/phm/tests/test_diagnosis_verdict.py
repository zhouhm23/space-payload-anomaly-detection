"""Tests for LLM diagnosis verdict parsing and auto-diagnosis mode."""

from __future__ import annotations

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from phm.services.diagnosis_service import DiagnosisService


class TestParseVerdict:
    """Tests for DiagnosisService._parse_verdict()."""

    def test_parse_real(self):
        text = "## 根因分析\n真实异常\nVERDICT: real"
        assert DiagnosisService._parse_verdict(text) == "real"

    def test_parse_false_alarm(self):
        text = "模型误报\nVERDICT: false_alarm"
        assert DiagnosisService._parse_verdict(text) == "false_alarm"

    def test_parse_uncertain(self):
        text = "VERDICT: uncertain"
        assert DiagnosisService._parse_verdict(text) == "uncertain"

    def test_parse_missing_returns_none(self):
        text = "没有 verdict 行的报告"
        assert DiagnosisService._parse_verdict(text) is None

    def test_parse_case_insensitive(self):
        text = "VERDICT: REAL"
        assert DiagnosisService._parse_verdict(text) == "real"

    def test_parse_extra_whitespace(self):
        text = "VERDICT:  real  \n"
        assert DiagnosisService._parse_verdict(text) == "real"

    def test_parse_invalid_value_returns_none(self):
        text = "VERDICT: maybe"
        assert DiagnosisService._parse_verdict(text) is None

    def test_parse_verdict_in_context(self):
        """Verdict line appears after a full report."""
        text = (
            "## 根因分析\n"
            "数据漂移导致传感器读数异常。\n\n"
            "## 趋势评估\n"
            "分数持续上升，建议关注。\n\n"
            "## 置信度\n"
            "中 — 需更多数据确认。\n"
            "VERDICT: uncertain"
        )
        assert DiagnosisService._parse_verdict(text) == "uncertain"


class TestAutoDiagnoseStatus:
    """Tests for auto_diagnose_all() status tracking."""

    def _make_disabled_service(self, warning_service=None, sqlite_store=None):
        return DiagnosisService(
            base_url="", api_key="", model="",
            warning_service=warning_service,
            sqlite_store=sqlite_store,
            config_service=None,
        )

    def test_auto_status_initial(self):
        svc = self._make_disabled_service()
        status = svc.auto_status
        assert status["running"] is False
        assert status["done"] == 0
        assert status["total"] == 0

    def test_auto_disabled_returns_error(self):
        svc = self._make_disabled_service(
            warning_service=types.SimpleNamespace(
                warnings=types.SimpleNamespace(all=lambda: []),
                list=lambda l: [],
            ),
        )
        result = svc.auto_diagnose_all()
        assert result["started"] is False
        assert result["error"] is not None
        assert "not configured" in result["error"].lower()


class TestJointAlertDiagnosis:
    """Joint-alert (alert_type=joint) diagnosis tests.

    A joint alert's channel is the virtual "SUB:<folder>"; it has no telemetry
    table / cascade. Context is extracted from alert_records.score_snapshot
    (a dict).
    """

    def _make_service_with_joint_alert(self, joint_alert):
        """Build a disabled DiagnosisService whose sqlite returns the given joint alert."""
        sqlite = types.SimpleNamespace(
            query_alerts=lambda limit=50: [joint_alert],
            get_diagnosis=lambda *a, **kw: None,
            save_diagnosis=lambda *a, **kw: None,
            update_alert_verdict=lambda *a, **kw: None,
            query_window=lambda *a, **kw: {"data": []},
        )
        return DiagnosisService(
            base_url="", api_key="", model="",
            warning_service=types.SimpleNamespace(
                warnings=types.SimpleNamespace(all=lambda: []),
                _latest_cascade={},
                _latest_predict_scores={},
            ),
            sqlite_store=sqlite,
            config_service=None,
        )

    def _sample_joint_alert(self):
        return {
            "id": 100,
            "channel": "SUB:数据集",
            "alert_type": "joint",
            "score": 0.72,
            "created_at": 1700000000.0,
            "score_snapshot": {
                "joint_curve": [0.3, 0.5, 0.72, 0.6, 0.4],
                "contributions": [
                    {"channel": "C-1", "score": 0.65, "threshold": 0.5, "over": True},
                    {"channel": "C-2", "score": 0.72, "threshold": 0.5, "over": True},
                ],
                "channels": ["C-1", "C-2"],
            },
        }

    def test_build_joint_context_returns_context(self):
        """A joint alert should successfully build a context (not return None)."""
        svc = self._make_service_with_joint_alert(self._sample_joint_alert())
        ctx = svc._build_context("SUB:数据集", "joint", alert_ts=1700000000.0)
        assert ctx is not None
        assert ctx.get("_joint") is True
        s = ctx["summary"]
        assert s["alert_type"] == "joint"
        assert s["joint_score"] == 0.72
        assert len(s["contributions"]) == 2
        assert s["sub_channels"] == ["C-1", "C-2"]

    def test_build_joint_context_none_when_alert_missing(self):
        """When no matching joint alert record is found, returns None."""
        svc = self._make_service_with_joint_alert(self._sample_joint_alert())
        # Use a non-existent alert_ts.
        ctx = svc._build_context("SUB:数据集", "joint", alert_ts=9999999999.0)
        assert ctx is None

    def test_diagnose_joint_no_no_detection_error(self):
        """A joint alert diagnosis must not report 'no detection data available'."""
        svc = self._make_service_with_joint_alert(self._sample_joint_alert())
        result = svc.diagnose("SUB:数据集", alert_type="joint",
                              alert_ts=1700000000.0, force_refresh=True)
        # Disabled service → no LLM call; error should be "not configured" rather
        # than "no detection data available".
        assert result["error"] != "no detection data available for channel SUB:数据集"
        assert "not configured" in (result.get("error") or "").lower()

    def test_build_joint_prompt_contains_contributions(self):
        """The joint prompt should include sub-channel contribution info."""
        svc = self._make_service_with_joint_alert(self._sample_joint_alert())
        ctx = svc._build_context("SUB:数据集", "joint", alert_ts=1700000000.0)
        prompt = svc._build_prompt("SUB:数据集", ctx)
        assert "联合告警" in prompt
        assert "C-1" in prompt
        assert "C-2" in prompt
        assert "0.72" in prompt  # joint_score

    def test_joint_prompt_does_not_mention_waveform(self):
        """The joint user prompt must not mention 'waveform' (joint alerts have
        no single-channel waveform)."""
        svc = self._make_service_with_joint_alert(self._sample_joint_alert())
        ctx = svc._build_context("SUB:数据集", "joint", alert_ts=1700000000.0)
        prompt = svc._build_prompt("SUB:数据集", ctx)
        # The joint prompt uses the "joint score curve" and must not contain
        # "raw waveform" or "waveform mini-chart".
        assert "波形" not in prompt
        assert "联合分数曲线" in prompt

    def test_joint_uses_joint_system_prompt(self):
        """Joint alerts must use _JOINT_SYSTEM_PROMPT (which does not mention
        the waveform mini-chart)."""
        from phm.services.diagnosis_service import _JOINT_SYSTEM_PROMPT, _SYSTEM_PROMPT
        # The joint system prompt must not contain "波形迷你图".
        assert "波形迷你图" not in _JOINT_SYSTEM_PROMPT
        # The generic system prompt DOES contain "波形迷你图" (control).
        assert "波形迷你图" in _SYSTEM_PROMPT
