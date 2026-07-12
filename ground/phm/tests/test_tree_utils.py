"""Unit tests for phm.services.tree_utils — pure recursive device-tree helpers.

Covers: recursive traversal, empty/flat/nested/deep trees, orphan sensors,
malformed nodes, aggregation-strategy fallback.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from phm.services.tree_utils import (  # noqa: E402
    get_aggregation_strategy,
    get_flat_sensors,
    get_folders,
    get_sensor_to_folder,
    get_sensors_in_folder,
)


# ---- test fixtures ----------------------------------------------------------

def _nested_tree():
    """Spec-shaped tree: 2 folders + 1 orphan sensor, with one empty folder."""
    return [
        {
            "id": "folder_power",
            "name": "电源模块",
            "type": "folder",
            "children": [
                {
                    "id": "s1",
                    "type": "sensor",
                    "name": "主控温度",
                    "sourceId": "file:NASA-MSL/C-1",
                    "channelName": "C-1",
                    "blockSize": 512,
                    "position": {"module": "电源模块", "x": 0.3, "y": 0.4},
                },
                {
                    "id": "s2",
                    "type": "sensor",
                    "name": "电压",
                    "sourceId": "file:NASA-MSL/M-1",
                    "channelName": "M-1",
                    "blockSize": 512,
                    "position": {"module": "电源模块", "x": 0.7, "y": 0.6},
                },
            ],
        },
        {
            "id": "folder_compute",
            "name": "计算模块",
            "type": "folder",
            "children": [
                {
                    "id": "s3",
                    "type": "sensor",
                    "name": "CPU温度",
                    "sourceId": "file:NASA-MSL/T-4",
                    "channelName": "T-4",
                    "blockSize": 512,
                    "position": {"module": "计算模块", "x": 0.5, "y": 0.5},
                },
            ],
        },
        # Orphan — not inside any folder
        {
            "id": "s4",
            "type": "sensor",
            "name": "环境温度",
            "sourceId": "file:NASA-MSL/C-2",
            "channelName": "C-2",
            "blockSize": 512,
            "position": {"x": 0.5, "y": 0.9},
        },
    ]


# ---- get_flat_sensors -------------------------------------------------------

class TestGetFlatSensors:
    def test_nested_tree_collects_all_three_groups(self):
        flat = get_flat_sensors(_nested_tree())
        ids = [n["id"] for n in flat]
        assert ids == ["s1", "s2", "s3", "s4"]  # DFS order

    def test_flat_legacy_config(self):
        """Old flat config (no folders) still yields all sensors."""
        tree = [
            {"id": "a", "type": "sensor", "sourceId": "file:X/A", "channelName": "A"},
            {"id": "b", "type": "sensor", "sourceId": "file:X/B", "channelName": "B"},
        ]
        flat = get_flat_sensors(tree)
        assert [n["id"] for n in flat] == ["a", "b"]

    def test_empty_tree(self):
        assert get_flat_sensors([]) == []

    def test_none_tree(self):
        assert get_flat_sensors(None) == []  # type: ignore[arg-type]

    def test_deeply_nested(self):
        tree = [
            {
                "id": "f1",
                "type": "folder",
                "children": [
                    {
                        "id": "f2",
                        "type": "folder",
                        "children": [
                            {"id": "deep", "type": "sensor", "sourceId": "file:X/D", "channelName": "D"},
                        ],
                    },
                    {"id": "shallow", "type": "sensor", "sourceId": "file:X/S", "channelName": "S"},
                ],
            },
        ]
        flat = get_flat_sensors(tree)
        assert [n["id"] for n in flat] == ["deep", "shallow"]

    def test_empty_folder_yields_nothing(self):
        tree = [{"id": "empty", "type": "folder", "children": []}]
        assert get_flat_sensors(tree) == []

    def test_malformed_nodes_skipped(self):
        tree = [None, "string", 42, {"id": "ok", "type": "sensor", "sourceId": "file:X/OK", "channelName": "OK"}]
        flat = get_flat_sensors(tree)  # type: ignore[arg-type]
        assert [n["id"] for n in flat] == ["ok"]

    def test_sensor_inferred_by_sourceid_without_type(self):
        """A node with sourceId but no type still counts as a sensor (backward compat)."""
        tree = [{"id": "n1", "sourceId": "file:X/A", "channelName": "A"}]
        flat = get_flat_sensors(tree)
        assert len(flat) == 1 and flat[0]["id"] == "n1"


# ---- get_folders ------------------------------------------------------------

class TestGetFolders:
    def test_nested_tree_returns_both_folders(self):
        folders = get_folders(_nested_tree())
        ids = [f["id"] for f in folders]
        assert ids == ["folder_power", "folder_compute"]

    def test_flat_config_no_folders(self):
        tree = [{"id": "a", "type": "sensor", "sourceId": "x"}]
        assert get_folders(tree) == []

    def test_empty_folder_still_returned(self):
        """A folder with explicit type but empty children IS returned — the UI
        needs to show empty folders so users can add sensors into them."""
        tree = [{"id": "f", "type": "folder", "children": []}]
        folders = get_folders(tree)
        assert len(folders) == 1 and folders[0]["id"] == "f"

    def test_folder_without_explicit_type(self):
        tree = [{"id": "f", "children": [{"id": "s", "type": "sensor", "sourceId": "x"}]}]
        folders = get_folders(tree)
        assert len(folders) == 1 and folders[0]["id"] == "f"


# ---- get_sensor_to_folder ---------------------------------------------------

class TestGetSensorToFolder:
    def test_only_foldered_sensors_mapped(self):
        mapping = get_sensor_to_folder(_nested_tree())
        assert mapping == {"C-1": "folder_power", "M-1": "folder_power", "T-4": "folder_compute"}

    def test_orphan_not_in_mapping(self):
        mapping = get_sensor_to_folder(_nested_tree())
        assert "C-2" not in mapping  # the orphan sensor is omitted

    def test_flat_config_empty_mapping(self):
        assert get_sensor_to_folder([{"id": "a", "type": "sensor", "sourceId": "x"}]) == {}


# ---- get_sensors_in_folder --------------------------------------------------

class TestGetSensorsInFolder:
    def test_power_folder_has_two_sensors(self):
        sensors = get_sensors_in_folder(_nested_tree(), "folder_power")
        assert [s["id"] for s in sensors] == ["s1", "s2"]

    def test_compute_folder_has_one_sensor(self):
        sensors = get_sensors_in_folder(_nested_tree(), "folder_compute")
        assert [s["id"] for s in sensors] == ["s3"]

    def test_unknown_folder_returns_empty(self):
        assert get_sensors_in_folder(_nested_tree(), "nonexistent") == []

    def test_orphan_id_returns_empty(self):
        """The orphan sensor s4 has no folder; asking by its id yields nothing."""
        assert get_sensors_in_folder(_nested_tree(), "s4") == []


# ---- get_aggregation_strategy -----------------------------------------------

class TestGetAggregationStrategy:
    def test_default_min_when_absent(self):
        assert get_aggregation_strategy({"device_tree": []}) == "min"

    def test_explicit_min(self):
        assert get_aggregation_strategy({"aggregation_strategy": "min"}) == "min"

    def test_explicit_mean(self):
        assert get_aggregation_strategy({"aggregation_strategy": "mean"}) == "mean"

    def test_invalid_falls_back_to_min(self):
        assert get_aggregation_strategy({"aggregation_strategy": "median"}) == "min"

    def test_none_falls_back_to_min(self):
        assert get_aggregation_strategy({}) == "min"
        assert get_aggregation_strategy(None) == "min"  # type: ignore[arg-type]
