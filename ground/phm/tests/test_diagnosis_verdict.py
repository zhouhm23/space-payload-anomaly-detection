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
