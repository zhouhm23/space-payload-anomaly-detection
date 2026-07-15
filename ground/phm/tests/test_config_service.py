"""Tests for ConfigService.save — especially the empty-tree safety guard.

These tests exist because the device_config.json was repeatedly wiped to an
empty tree during acceptance testing (front-end delete-all + auto-save wrote
an empty tree to disk, stopping all collection).  The safety guard in
ConfigService.save refuses to persist an empty tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phm.services.config_service import ConfigService


@pytest.fixture
def cfg_service(tmp_path: Path) -> ConfigService:
    """A ConfigService backed by a temp JSON file (no real space-segment TCP)."""
    return ConfigService(tmp_path / "device_config.json", space_host="127.0.0.1", space_port=19999)


@pytest.fixture
def cfg_service_with_data(cfg_service: ConfigService) -> ConfigService:
    """Pre-seed the config file with a non-empty tree."""
    cfg_service.config_path.write_text(
        json.dumps({
            "device_tree": [
                {"id": "n1", "name": "C-1", "type": "sensor",
                 "sourceId": "file:NASA-MSL/C-1", "channelName": "C-1", "blockSize": 512},
            ],
            "aggregation_strategy": "min",
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return cfg_service


class TestConfigServiceSave:
    """ConfigService.save edge-case coverage."""

    def test_save_normal_tree(self, cfg_service: ConfigService):
        """A non-empty tree is saved successfully."""
        tree = [{"id": "n1", "name": "S1", "type": "sensor",
                 "sourceId": "virtual:sine", "channelName": "VS-sine", "blockSize": 512}]
        result = cfg_service.save({"device_tree": tree})
        assert result["status"] == "ok"
        saved = json.loads(cfg_service.config_path.read_text(encoding="utf-8"))
        assert len(saved["device_tree"]) == 1
        assert saved["aggregation_strategy"] == "min"

    def test_save_empty_tree_refused(self, cfg_service_with_data: ConfigService):
        """Saving an empty device_tree must be refused — this is the guard
        that prevents the recurring 'config wiped to empty' bug."""
        result = cfg_service_with_data.save({"device_tree": []})
        assert result["status"] == "error"
        assert "空设备树" in result["message"]
        # The existing config on disk must be untouched.
        on_disk = json.loads(cfg_service_with_data.config_path.read_text(encoding="utf-8"))
        assert len(on_disk["device_tree"]) == 1, "existing config must not be wiped"

    def test_save_missing_device_tree_key_refused(self, cfg_service_with_data: ConfigService):
        """A POST body with no device_tree key is also treated as empty."""
        result = cfg_service_with_data.save({"aggregation_strategy": "min"})
        assert result["status"] == "error"

    def test_save_duplicate_source_refused(self, cfg_service: ConfigService):
        """Duplicate sourceId is rejected (two sensors sharing one channel)."""
        tree = [
            {"id": "n1", "name": "A", "type": "sensor", "sourceId": "virtual:sine", "channelName": "VS-sine"},
            {"id": "n2", "name": "B", "type": "sensor", "sourceId": "virtual:sine", "channelName": "VS-sine"},
        ]
        result = cfg_service.save({"device_tree": tree})
        assert result["status"] == "error"
        assert "重复" in result["message"]

    def test_load_returns_empty_tree_when_file_missing(self, cfg_service: ConfigService):
        """When the config file does not exist, load() returns a default skeleton."""
        data = cfg_service.load()
        assert data["device_tree"] == []
        assert data["aggregation_strategy"] == "min"
