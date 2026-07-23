"""Unit tests for SystemConfigService / ThemeService admin save capability.

Django-free (pure Python) tests covering:
  - raw_with_docs: returns the on-disk original including _doc keys
  - save: type validation / unknown section/key rejection /
    disk write + hot reload
  - _doc field preservation: raw_with_docs still shows _doc after save
  - readonly-key rejection (SystemConfigService.llm.timeout_sec)
  - nested-object key rejection (ThemeService.chart.padding)

Uses tmp_path to isolate the real JSON and avoid polluting repo config.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phm.services.system_config_service import (
    SystemConfigService, reset_system_config,
)
from phm.services.theme_service import ThemeService, reset_theme


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_system_config(tmp_path: Path) -> Path:
    """Build a complete system_config.json (including _doc) under tmp_path."""
    data = {
        "_doc": "top-level doc",
        "thresholds": {
            "_doc": "thresholds section doc",
            "anomaly": 0.5,
            "l1_sigma_k": 3.0,
        },
        "storage": {
            "_doc": "storage section doc",
            "sqlite_enabled": True,
            "ring_buffer_max": 20000,
            "db_path": "data/phm.db",
        },
        "llm": {
            "_doc": "llm section doc",
            "timeout_sec": 30.0,
        },
    }
    p = tmp_path / "system_config.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def tmp_theme(tmp_path: Path) -> Path:
    data = {
        "_doc": "top-level theme doc",
        "colors": {
            "_doc": "colors doc",
            "bgPrimary": "#0b0f1a", "blue": "#2d8cf0",
        },
        "poll": {
            "_doc": "poll doc",
            "chart": 2000, "health": 3000,
        },
        "chart": {
            "_doc": "chart doc",
            "cacheCount": 2048,
            "padding": {"top": 20, "right": 50},
        },
    }
    p = tmp_path / "ui_theme.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ── SystemConfigService ──────────────────────────────────────────────────

class TestSystemConfigSave:
    def test_raw_with_docs_includes_doc_keys(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        raw = svc.raw_with_docs()
        assert raw["_doc"] == "top-level doc"
        assert raw["thresholds"]["_doc"] == "thresholds section doc"

    def test_save_updates_runtime_value(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        result = svc.save("thresholds", "anomaly", 0.42)
        assert result["status"] == "ok"
        assert result["old"] == 0.5
        assert result["new"] == 0.42
        # Hot reload: after load() the attribute is in sync.
        assert svc.thresholds["anomaly"] == 0.42

    def test_save_preserves_doc(self, tmp_system_config):
        """save must not drop _doc."""
        svc = SystemConfigService(str(tmp_system_config))
        svc.save("thresholds", "anomaly", 0.42)
        raw = svc.raw_with_docs()
        assert raw["thresholds"]["_doc"] == "thresholds section doc"
        assert raw["_doc"] == "top-level doc"

    def test_save_rejects_unknown_section(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        r = svc.save("nonexistent", "x", 1)
        assert r["status"] == "error"
        assert "未知配置项" in r["message"]

    def test_save_rejects_unknown_key(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        r = svc.save("thresholds", "nonexistent", 1)
        assert r["status"] == "error"
        assert "未知配置项" in r["message"]

    def test_save_rejects_type_mismatch(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        # anomaly expects float; pass a str.
        r = svc.save("thresholds", "anomaly", "high")
        assert r["status"] == "error"
        assert "期望 float" in r["message"]

    def test_save_rejects_bool_for_int(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        # ring_buffer_max expects int; bool should be rejected (avoid True == 1 mis-acceptance).
        r = svc.save("storage", "ring_buffer_max", True)
        assert r["status"] == "error"

    def test_save_rejects_readonly_key(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        # llm.timeout_sec is flagged read-only.
        r = svc.save("llm", "timeout_sec", 60.0)
        assert r["status"] == "error"
        assert "只读" in r["message"]

    def test_save_persists_to_disk(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        svc.save("storage", "ring_buffer_max", 40000)
        # Verify by reading the file directly.
        on_disk = json.loads(tmp_system_config.read_text(encoding="utf-8"))
        assert on_disk["storage"]["ring_buffer_max"] == 40000
        # _doc is still present.
        assert on_disk["storage"]["_doc"] == "storage section doc"

    def test_save_invalid_section_type(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        r = svc.save(123, "x", 1)  # type: ignore[arg-type]
        assert r["status"] == "error"

    def test_is_readonly_flag(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        assert svc.is_readonly("llm", "timeout_sec") is True
        assert svc.is_readonly("thresholds", "anomaly") is False

    def test_display_names_returns_dict(self, tmp_system_config):
        svc = SystemConfigService(str(tmp_system_config))
        names = svc.display_names()
        assert "thresholds" in names
        assert "anomaly" in names["thresholds"]


# ── ThemeService ─────────────────────────────────────────────────────────

class TestThemeServiceSave:
    def test_raw_with_docs_includes_doc(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        raw = svc.raw_with_docs()
        assert raw["_doc"] == "top-level theme doc"
        assert raw["colors"]["_doc"] == "colors doc"

    def test_save_updates_runtime(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        r = svc.save("colors", "blue", "#abcdef")
        assert r["status"] == "ok"
        assert r["old"] == "#2d8cf0"
        assert r["new"] == "#abcdef"
        # as_dict reflects the new value.
        assert svc.as_dict()["colors"]["blue"] == "#abcdef"

    def test_save_preserves_doc(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        svc.save("colors", "blue", "#abcdef")
        raw = svc.raw_with_docs()
        assert raw["colors"]["_doc"] == "colors doc"

    def test_save_rejects_nested_key(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        # chart.padding is a nested object; should be rejected.
        r = svc.save("chart", "padding", {"top": 99})
        assert r["status"] == "error"

    def test_save_rejects_type_mismatch(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        # chart.cacheCount expects int; pass a str.
        r = svc.save("chart", "cacheCount", "lots")
        assert r["status"] == "error"

    def test_save_rejects_unknown_key(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        r = svc.save("colors", "nonexistent", "#000")
        assert r["status"] == "error"

    def test_save_persists_to_disk(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        svc.save("poll", "chart", 5000)
        on_disk = json.loads(tmp_theme.read_text(encoding="utf-8"))
        assert on_disk["poll"]["chart"] == 5000

    def test_is_readonly_for_nested_parent(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        # chart.padding is read-only as a whole.
        assert svc.is_readonly("chart", "padding") is True
        # chart.cacheCount is a scalar → editable.
        assert svc.is_readonly("chart", "cacheCount") is False

    def test_display_names_returns_dict(self, tmp_theme):
        svc = ThemeService(str(tmp_theme))
        names = svc.display_names()
        assert "colors" in names
        assert "blue" in names["colors"]
