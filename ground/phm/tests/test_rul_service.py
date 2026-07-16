"""Tests for the RUL service layer.

Covers:
- ``@rul:fd00X`` tag parsing from sensor descriptions (case-insensitive,
  missing description, no tag).
- :class:`RulService.channels_with_rul` device-tree scan.
- :class:`CMAPSSDataSource` playback (channels, window shape, advance loops).
- :class:`RulService.predict_all` end-to-end with a mock predictor (advance
  called, history accumulated, result fields complete).

The mock predictor avoids loading the real LSTM weights so the suite stays
fast (<1s) and runs without the C-MAPSS dataset present.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from phm.services.config_service import ConfigService
from phm.services.rul_service import (
    CMAPSSDataSource,
    RulService,
    parse_rul_tag,
)


# ---------------------------------------------------------------------------
# parse_rul_tag
# ---------------------------------------------------------------------------

class TestParseRulTag:
    def test_lowercase(self):
        assert parse_rul_tag("text @rul:fd001 more text") == "fd001"

    def test_uppercase(self):
        assert parse_rul_tag("@rul:FD003 end") == "fd003"

    def test_mixed_case(self):
        assert parse_rul_tag("@rul:Fd002") == "fd002"

    def test_with_whitespace(self):
        assert parse_rul_tag("@rul:  fd004") == "fd004"

    def test_no_tag(self):
        assert parse_rul_tag("plain description, no tag") is None

    def test_empty(self):
        assert parse_rul_tag("") is None

    def test_none(self):
        assert parse_rul_tag(None) is None

    def test_first_match_wins(self):
        # If two tags are present, the first one wins (deterministic).
        assert parse_rul_tag("@rul:fd001 @rul:fd003") == "fd001"

    def test_invalid_subset_ignored(self):
        # fd005 is not a valid subset (regex only matches fd00[1234]).
        assert parse_rul_tag("@rul:fd005") is None


# ---------------------------------------------------------------------------
# RulService.channels_with_rul
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_service(tmp_path: Path) -> ConfigService:
    return ConfigService(tmp_path / "device_config.json",
                         space_host="127.0.0.1", space_port=19999)


def _make_predictors(model_ids: list[str]) -> dict:
    """Build mock predictors keyed by model id.

    Each mock returns a fixed RUL and max_rul=125 so tests can assert without
    loading real weights.
    """
    preds = {}
    for mid in model_ids:
        m = MagicMock()
        m.max_rul = 125
        m.predict_rul.return_value = 100.0
        preds[mid] = m
    return preds


def _wrap_source(source, model_id: str = "fd001") -> dict:
    """Wrap a single mock source into the ``{model_id: source}`` dict that
    RulService now expects (multi-source routing)."""
    return {model_id: source}


class TestChannelsWithRul:
    def test_finds_tagged_sensor(self, cfg_service: ConfigService):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "CMAPSS_FD001_1",
                 "description": "demo @rul:fd001"},
            ],
        }), encoding="utf-8")
        svc = RulService(
            data_sources=_wrap_source(MagicMock()),
            predictors=_make_predictors(["fd001"]),
            config_service=cfg_service,
        )
        assert svc.channels_with_rul() == {"CMAPSS_FD001_1": "fd001"}

    def test_skips_untagged_sensor(self, cfg_service: ConfigService):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "C-1",
                 "description": "MSL channel, no RUL"},
            ],
        }), encoding="utf-8")
        svc = RulService(_wrap_source(MagicMock()), _make_predictors(["fd001"]), cfg_service)
        assert svc.channels_with_rul() == {}

    def test_skips_tag_with_unloaded_model(self, cfg_service: ConfigService):
        # Tag says fd002 but only fd001 predictor is loaded → skip.
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "X",
                 "description": "@rul:fd002"},
            ],
        }), encoding="utf-8")
        svc = RulService(_wrap_source(MagicMock()), _make_predictors(["fd001"]), cfg_service)
        assert svc.channels_with_rul() == {}

    def test_nested_folder(self, cfg_service: ConfigService):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "folder", "name": "engines", "children": [
                    {"type": "sensor", "channelName": "E1",
                     "description": "@rul:fd001"},
                    {"type": "sensor", "channelName": "E2",
                     "description": "no tag"},
                ]},
            ],
        }), encoding="utf-8")
        svc = RulService(_wrap_source(MagicMock()), _make_predictors(["fd001"]), cfg_service)
        assert svc.channels_with_rul() == {"E1": "fd001"}

    def test_falls_back_to_name_when_no_channelName(self, cfg_service: ConfigService):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "name": "FallbackName",
                 "description": "@rul:fd001"},
            ],
        }), encoding="utf-8")
        svc = RulService(_wrap_source(MagicMock()), _make_predictors(["fd001"]), cfg_service)
        assert svc.channels_with_rul() == {"FallbackName": "fd001"}


# ---------------------------------------------------------------------------
# CMAPSSDataSource (uses the real dataset if present, else skips)
# ---------------------------------------------------------------------------

_CMAPSS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "datasets" / "CMAPSSData"


@pytest.mark.skipif(
    not (_CMAPSS_DIR / "test_FD001.txt").exists(),
    reason="C-MAPSS dataset not present",
)
class TestCMAPSSDataSource:
    def test_channels_count(self):
        ds = CMAPSSDataSource(_CMAPSS_DIR, subset="FD001")
        chs = ds.channels()
        assert len(chs) == 100
        assert chs[0] == "CMAPSS_FD001_1"
        assert chs[-1] == "CMAPSS_FD001_100"

    def test_window_shape(self):
        ds = CMAPSSDataSource(_CMAPSS_DIR, subset="FD001")
        w = ds.get_window("CMAPSS_FD001_1", 30)
        assert w is not None
        assert w.shape == (30, 14)
        assert w.dtype == np.float32

    def test_unknown_channel_returns_none(self):
        ds = CMAPSSDataSource(_CMAPSS_DIR, subset="FD001")
        assert ds.get_window("NOPE", 30) is None

    def test_advance_progresses_pointer(self):
        ds = CMAPSSDataSource(_CMAPSS_DIR, subset="FD001")
        # After init pointer=1, so window is the first cycle padded to 30.
        w1 = ds.get_window("CMAPSS_FD001_1", 5)
        ds.advance()  # pointer → 2
        w2 = ds.get_window("CMAPSS_FD001_1", 5)
        # The 2nd window's last row should be the 2nd cycle (new data).
        assert not np.allclose(w1[-1], w2[-1])

    def test_advance_loops_back(self):
        """When an engine is exhausted the pointer resets to 1 (demo loops)."""
        ds = CMAPSSDataSource(_CMAPSS_DIR, subset="FD001")
        # Force pointer to the end by advancing many times (FD001 engines have
        # ~200-300 cycles; advance 500x guarantees wraparound).
        for _ in range(500):
            ds.advance()
        # Should still serve a valid window (not crash, not return None).
        w = ds.get_window("CMAPSS_FD001_1", 30)
        assert w is not None
        assert w.shape == (30, 14)


# ---------------------------------------------------------------------------
# RulService.predict_all / predict
# ---------------------------------------------------------------------------

class TestRulServicePredict:
    def _build_svc(
        self, cfg_service: ConfigService, channels: list[str]
    ) -> RulService:
        """Build a RulService with a mock data source serving ``channels``."""
        source = MagicMock()
        source.channels.return_value = channels
        # Serve a deterministic (30,14) window for any channel.
        source.get_window.return_value = np.zeros((30, 14), dtype=np.float32)
        preds = _make_predictors(["fd001"])
        return RulService(_wrap_source(source), preds, cfg_service)

    def test_predict_all_returns_one_result_per_tagged_channel(
        self, cfg_service: ConfigService
    ):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "E1", "description": "@rul:fd001"},
                {"type": "sensor", "channelName": "E2", "description": "@rul:fd001"},
            ],
        }), encoding="utf-8")
        svc = self._build_svc(cfg_service, ["E1", "E2"])
        results = svc.predict_all()
        assert len(results) == 2
        names = {r["channel"] for r in results}
        assert names == {"E1", "E2"}

    def test_predict_all_result_fields(self, cfg_service: ConfigService):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "E1", "description": "@rul:fd001"},
            ],
        }), encoding="utf-8")
        svc = self._build_svc(cfg_service, ["E1"])
        results = svc.predict_all()
        r = results[0]
        for key in ("channel", "rul", "max_rul", "unit", "model", "source", "history"):
            assert key in r, f"missing field {key}"
        assert r["channel"] == "E1"
        assert r["model"] == "fd001"
        assert r["max_rul"] == 125
        assert r["unit"] == "cycles"
        assert r["rul"] == 100.0

    def test_advance_called_once_per_predict_all(self, cfg_service: ConfigService):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "E1", "description": "@rul:fd001"},
            ],
        }), encoding="utf-8")
        svc = self._build_svc(cfg_service, ["E1"])
        svc.predict_all()
        svc.predict_all()
        assert svc._sources["fd001"].advance.call_count == 2

    def test_history_accumulates_and_caps(self, cfg_service: ConfigService):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "E1", "description": "@rul:fd001"},
            ],
        }), encoding="utf-8")
        svc = self._build_svc(cfg_service, ["E1"])
        svc.history_len = 3  # small cap for the test
        svc.predict_all()
        svc.predict_all()
        svc.predict_all()
        svc.predict_all()  # 4th call → history should still be length 3
        assert len(svc._history["E1"]) == 3

    def test_predict_single_channel_no_advance(self, cfg_service: ConfigService):
        """predict() (single) must NOT advance the data source pointer."""
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "E1", "description": "@rul:fd001"},
            ],
        }), encoding="utf-8")
        svc = self._build_svc(cfg_service, ["E1"])
        r = svc.predict("E1")
        assert r is not None
        assert r["channel"] == "E1"
        assert svc._sources["fd001"].advance.call_count == 0

    def test_predict_unknown_channel_returns_none(self, cfg_service: ConfigService):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "E1", "description": "@rul:fd001"},
            ],
        }), encoding="utf-8")
        svc = self._build_svc(cfg_service, ["E1"])
        assert svc.predict("NOPE") is None

    def test_predict_all_skips_channel_when_source_returns_none(
        self, cfg_service: ConfigService
    ):
        cfg_service.config_path.write_text(json.dumps({
            "device_tree": [
                {"type": "sensor", "channelName": "E1", "description": "@rul:fd001"},
                {"type": "sensor", "channelName": "E2", "description": "@rul:fd001"},
            ],
        }), encoding="utf-8")
        source = MagicMock()
        source.channels.return_value = ["E1", "E2"]
        # E2 has no data yet → skipped.
        source.get_window.side_effect = lambda ch, n: (
            np.zeros((30, 14), dtype=np.float32) if ch == "E1" else None
        )
        svc = RulService(_wrap_source(source), _make_predictors(["fd001"]), cfg_service)
        results = svc.predict_all()
        assert len(results) == 1
        assert results[0]["channel"] == "E1"
