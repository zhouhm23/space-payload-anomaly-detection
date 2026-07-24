"""Unit tests for the ``@command`` DSL (Plan 3, v2 rewrite).

Coverage mirrors the verification matrix in the plan document:

  * Parser — 15+ cases covering empty / single / multi-command / mixed
    prose / unknown tokens / param paths.
  * Validator — every hard error (E1-E5) and warning (W1-W2) with at
    least two cases each.
  * Calibrator — three-tier parameter precedence (DSL > offline > default)
    and per-mode setpoint translation.
  * Persistence round-trip — description → parse → validate →
    to_calibration → upsert → reload → field equality.

These tests do NOT exercise the Django view hook (that is covered in
``test_views_admin_device_tree.py``); they operate purely on the
algorithm-layer package.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "ground"))

from phm.algorithm.sensor_dsl import (
    parse,
    validate,
    to_calibration,
    classify_layer,
    SensorConfig,
)
from phm.algorithm.sensor_dsl.commands import COMMANDS
from phm.algorithm.calibration_config import (
    CalibrationConfig,
    ChannelCalibration,
)
from phm.algorithm.rules import (
    FILTER_REGISTRY,
    DEFAULT_L1_MODULES,
    DEFAULT_L3_MODULES,
    build_filter,
)
from phm.algorithm._registry import MODEL_REGISTRY


# ════════════════════════════════════════════════════════════════════════════
# Parser
# ════════════════════════════════════════════════════════════════════════════

class TestParserBasics:
    """Pure parsing — never raises, never judges legality."""

    def test_none_returns_empty_config(self):
        cfg = parse(None)
        assert cfg.is_empty
        assert cfg.algorithms == []
        assert cfg.threshold is None

    def test_empty_string_returns_empty_config(self):
        assert parse("").is_empty

    def test_whitespace_only_returns_empty_config(self):
        assert parse("   \n\t  ").is_empty

    def test_pure_prose_no_commands(self):
        cfg = parse("恒温室传感器，温度25度")
        assert cfg.is_empty
        # Prose must not be misread as a token.
        assert cfg.raw_tokens == []

    def test_single_algorithm(self):
        cfg = parse("@算法=tspulse")
        assert cfg.algorithms == ["tspulse"]

    def test_multiple_comma_algorithms(self):
        cfg = parse("@算法=l1_setpoint,tspulse,l3_range")
        assert cfg.algorithms == ["l1_setpoint", "tspulse", "l3_range"]

    def test_skip_detector_flag(self):
        cfg = parse("@算法=l1_setpoint @跳过模型")
        assert cfg.skip_detector is True
        assert cfg.algorithms == ["l1_setpoint"]

    def test_threshold_float(self):
        cfg = parse("@阈值=0.7")
        assert cfg.threshold == pytest.approx(0.7)

    def test_threshold_integer(self):
        cfg = parse("@阈值=0")
        assert cfg.threshold == 0.0

    def test_threshold_zero_decimal(self):
        cfg = parse("@阈值=0.5")
        assert cfg.threshold == 0.5

    def test_param_path_command_value(self):
        cfg = parse("@参数.l1_setpoint.常态值=0")
        assert cfg.params == {"l1_setpoint": {"常态值": 0.0}}

    def test_param_path_anomaly_list(self):
        cfg = parse("@参数.l1_setpoint.异常值=1,2,3")
        assert cfg.params["l1_setpoint"]["异常值"] == "1,2,3"

    def test_multiple_param_paths_same_module(self):
        cfg = parse(
            "@参数.l1_setpoint.期望值=25 @参数.l1_setpoint.容差=2"
        )
        assert cfg.params["l1_setpoint"] == {"期望值": 25.0, "容差": 2.0}

    def test_multiple_param_paths_different_modules(self):
        cfg = parse(
            "@参数.l1_setpoint.期望值=25 @参数.l1_sigma.sigma_k=4.0"
        )
        assert set(cfg.params.keys()) == {"l1_setpoint", "l1_sigma"}

    def test_unknown_at_token_is_prose(self):
        # ``@备注=foo`` is not in COMMANDS — must be silently ignored.
        cfg = parse("@备注=hello world @算法=tspulse")
        assert cfg.algorithms == ["tspulse"]
        # raw_tokens still records the unknown token for diagnostics.
        names = [t[0] for t in cfg.raw_tokens]
        assert "备注" in names

    def test_mixed_prose_and_commands(self):
        cfg = parse(
            "电池电压，1V→5V充电5min，1min耗电 @算法=tspulse @阈值=0.6"
        )
        assert cfg.algorithms == ["tspulse"]
        assert cfg.threshold == pytest.approx(0.6)

    def test_adjacent_tokens_do_not_merge(self):
        # Two param tokens separated by a single space — the regex must
        # treat them as two tokens, not one big value.
        cfg = parse("@参数.x.a=0 @参数.y.b=1")
        assert cfg.params == {"x": {"a": 0.0}, "y": {"b": 1.0}}

    def test_trailing_comma_in_algorithm(self):
        # Defensive: trailing comma shouldn't produce an empty name.
        cfg = parse("@算法=tspulse,")
        assert cfg.algorithms == ["tspulse"]

    def test_command_with_chinese_full_width_chars_around(self):
        # Full-width punctuation *adjacent* to a token's value is part of
        # the value (the regex charset is "non-space non-@ non-=").  The
        # scientist is expected to leave a space before any trailing
        # punctuation, which this test reflects.
        cfg = parse("（@算法=tspulse ）")
        assert cfg.algorithms == ["tspulse"]

    def test_threshold_non_numeric_kept_as_string(self):
        # Parser never raises; bad numeric is passed through as string so
        # the validator can produce a precise error.
        cfg = parse("@阈值=abc")
        assert cfg.threshold == "abc"


# ════════════════════════════════════════════════════════════════════════════
# Validator — hard errors E1-E5
# ════════════════════════════════════════════════════════════════════════════

class TestValidatorE1UnknownAlgorithm:
    """E1: every @算法= name must be in FILTER_REGISTRY ∪ MODEL_REGISTRY."""

    def test_completely_unknown_name(self):
        cfg = parse("@算法=does_not_exist")
        errs, _ = validate(cfg)
        assert any("E1" in e and "does_not_exist" in e for e in errs)

    def test_one_unknown_among_known(self):
        cfg = parse("@算法=tspulse,fake_algo")
        errs, _ = validate(cfg)
        assert any("E1" in e and "fake_algo" in e for e in errs)
        # The known one must NOT trigger E1.
        assert not any("tspulse" in e and "E1" in e for e in errs)

    def test_known_filter_passes_e1(self):
        cfg = parse("@算法=l1_sigma")
        errs, _ = validate(cfg)
        assert not any("E1" in e for e in errs)

    def test_known_model_passes_e1(self):
        cfg = parse("@算法=tspulse")
        errs, _ = validate(cfg)
        assert not any("E1" in e for e in errs)


class TestValidatorE2SkipModelMutex:
    """E2: @跳过模型 cannot coexist with an L2 detector in @算法=."""

    def test_skip_with_detector_is_error(self):
        cfg = parse("@算法=tspulse @跳过模型")
        errs, _ = validate(cfg)
        assert any("E2" in e for e in errs)

    def test_skip_without_detector_ok(self):
        cfg = parse("@算法=l1_setpoint @跳过模型 "
                    "@参数.l1_setpoint.常态值=0 @参数.l1_setpoint.异常值=1")
        errs, _ = validate(cfg)
        assert not any("E2" in e for e in errs)

    def test_detector_without_skip_ok(self):
        cfg = parse("@算法=tspulse")
        errs, _ = validate(cfg)
        assert not any("E2" in e for e in errs)


class TestValidatorE3SetpointAnchor:
    """E3: l1_setpoint needs at least one anchor parameter."""

    def test_setpoint_without_any_param_is_error(self):
        cfg = parse("@算法=l1_setpoint")
        errs, _ = validate(cfg)
        assert any("E3" in e for e in errs)

    def test_setpoint_with_command_anchor_ok(self):
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.常态值=0 "
                    "@参数.l1_setpoint.异常值=1")
        errs, _ = validate(cfg)
        assert not any("E3" in e for e in errs)

    def test_setpoint_with_range_anchor_ok(self):
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.期望值=25")
        errs, _ = validate(cfg)
        assert not any("E3" in e for e in errs)

    def test_setpoint_with_range_low_high_ok(self):
        cfg = parse("@算法=l1_setpoint "
                    "@参数.l1_setpoint.范围下限=20 @参数.l1_setpoint.范围上限=30")
        errs, _ = validate(cfg)
        assert not any("E3" in e for e in errs)

    def test_setpoint_with_enumerate_anchor_ok(self):
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.合法值=0,1,2,3")
        errs, _ = validate(cfg)
        assert not any("E3" in e for e in errs)

    def test_setpoint_with_unrelated_param_still_error(self):
        # A non-anchor key on l1_setpoint doesn't satisfy E3.
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.foo=1")
        errs, _ = validate(cfg)
        assert any("E3" in e for e in errs)


class TestValidatorE4ThresholdRange:
    """E4: @阈值= must be a number in [0, 1]."""

    def test_threshold_above_one(self):
        cfg = parse("@阈值=1.5")
        errs, _ = validate(cfg)
        assert any("E4" in e for e in errs)

    def test_threshold_negative(self):
        cfg = parse("@阈值=-0.1")
        errs, _ = validate(cfg)
        assert any("E4" in e for e in errs)

    def test_threshold_non_numeric(self):
        cfg = parse("@阈值=abc")
        errs, _ = validate(cfg)
        assert any("E4" in e for e in errs)

    def test_threshold_zero_ok(self):
        cfg = parse("@阈值=0")
        errs, _ = validate(cfg)
        assert not any("E4" in e for e in errs)

    def test_threshold_one_ok(self):
        cfg = parse("@阈值=1")
        errs, _ = validate(cfg)
        assert not any("E4" in e for e in errs)

    def test_threshold_half_ok(self):
        cfg = parse("@阈值=0.5")
        errs, _ = validate(cfg)
        assert not any("E4" in e for e in errs)


class TestValidatorE5ParamModuleDeclared:
    """E5: @参数.<module>.* requires <module> to appear in @算法=."""

    def test_param_for_undeclared_module(self):
        cfg = parse("@算法=tspulse @参数.l1_sigma.sigma_k=4.0")
        errs, _ = validate(cfg)
        assert any("E5" in e and "l1_sigma" in e for e in errs)

    def test_param_for_declared_module_ok(self):
        cfg = parse("@算法=l1_sigma @参数.l1_sigma.sigma_k=4.0")
        errs, _ = validate(cfg)
        assert not any("E5" in e for e in errs)

    def test_param_for_setpoint_when_declared_ok(self):
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.期望值=25")
        errs, _ = validate(cfg)
        assert not any("E5" in e for e in errs)


# ════════════════════════════════════════════════════════════════════════════
# Validator — warnings W1-W2
# ════════════════════════════════════════════════════════════════════════════

class TestValidatorWarnings:
    """W1: uncovered layer → default; W2: no commands → full default."""

    def test_empty_description_warns_full_default(self):
        cfg = parse("")
        errs, warns = validate(cfg)
        assert errs == []
        assert any("W2" in w or "系统默认全流程" in w for w in warns)

    def test_prose_only_warns_full_default(self):
        cfg = parse("just prose, no commands")
        errs, warns = validate(cfg)
        assert errs == []
        assert any("系统默认" in w for w in warns)

    def test_only_l2_declared_warns_l1_l3_default(self):
        cfg = parse("@算法=tspulse")
        _, warns = validate(cfg)
        # Both L1 and L3 default warnings should fire.
        assert any("L1" in w for w in warns)
        assert any("L3" in w for w in warns)

    def test_skip_model_warns_l2_skipped(self):
        cfg = parse("@算法=l1_setpoint @跳过模型 "
                    "@参数.l1_setpoint.常态值=0 @参数.l1_setpoint.异常值=1")
        _, warns = validate(cfg)
        assert any("L2" in w and "跳过" in w for w in warns)

    def test_fully_explicit_no_warnings(self):
        # Cover all three layers explicitly.
        cfg = parse("@算法=l1_sigma,tspulse,l3_range")
        errs, warns = validate(cfg)
        assert errs == []
        # No layer is defaulting.
        assert not any("默认" in w for w in warns)


# ════════════════════════════════════════════════════════════════════════════
# Layer classification
# ════════════════════════════════════════════════════════════════════════════

class TestClassifyLayer:
    """classify_layer: name → cascade layer."""

    def test_l1_prefix(self):
        assert classify_layer("l1_sigma") == "L1"
        assert classify_layer("l1_setpoint") == "L1"

    def test_l3_prefix(self):
        assert classify_layer("l3_range") == "L3"

    def test_detector_model(self):
        assert classify_layer("tspulse") == "L2"

    def test_forecaster_is_none(self):
        # ttm_r3 is a forecaster — not a cascade layer.
        assert classify_layer("ttm_r3") is None

    def test_rul_is_none(self):
        assert classify_layer("rul") is None

    def test_unknown_is_none(self):
        assert classify_layer("totally_unknown") is None


# ════════════════════════════════════════════════════════════════════════════
# Calibrator
# ════════════════════════════════════════════════════════════════════════════

class TestCalibratorLayerSplit:
    """to_calibration routes algorithm names to the right module list."""

    def test_l1_goes_to_l1_modules(self):
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.期望值=25")
        cal = to_calibration(cfg, None)
        assert cal.l1_modules == ["l1_setpoint"]
        assert cal.l3_modules is None

    def test_l3_goes_to_l3_modules(self):
        cfg = parse("@算法=l3_range")
        cal = to_calibration(cfg, None)
        assert cal.l3_modules == ["l3_range"]
        assert cal.l1_modules is None

    def test_detector_goes_to_detector_model(self):
        cfg = parse("@算法=tspulse")
        cal = to_calibration(cfg, None)
        assert cal.detector_model == "tspulse"
        assert cal.skip_detector is False

    def test_skip_detector_clears_detector_model(self):
        cfg = parse("@算法=l1_setpoint @跳过模型 "
                    "@参数.l1_setpoint.常态值=0 @参数.l1_setpoint.异常值=1")
        cal = to_calibration(cfg, None)
        assert cal.skip_detector is True
        assert cal.detector_model is None


class TestCalibratorSetpointTranslation:
    """to_calibration translates Chinese DSL keys to Python kwargs + mode."""

    def test_command_mode_translation(self):
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.常态值=0 "
                    "@参数.l1_setpoint.异常值=1,2")
        cal = to_calibration(cfg, None)
        sp = cal.module_params["l1_setpoint"]
        assert sp["mode"] == "command"
        assert sp["command_value"] == 0.0
        assert sp["anomaly_values"] == [1.0, 2.0]

    def test_range_mode_with_expected_tolerance(self):
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.期望值=25 "
                    "@参数.l1_setpoint.容差=2")
        cal = to_calibration(cfg, None)
        sp = cal.module_params["l1_setpoint"]
        assert sp["mode"] == "range"
        assert sp["expected"] == 25.0
        assert sp["tolerance"] == 2.0

    def test_range_mode_with_explicit_band(self):
        cfg = parse("@算法=l1_setpoint "
                    "@参数.l1_setpoint.范围下限=20 @参数.l1_setpoint.范围上限=30")
        cal = to_calibration(cfg, None)
        sp = cal.module_params["l1_setpoint"]
        assert sp["mode"] == "range"
        assert sp["range_low"] == 20.0
        assert sp["range_high"] == 30.0

    def test_enumerate_mode(self):
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.合法值=0,1,2,3")
        cal = to_calibration(cfg, None)
        sp = cal.module_params["l1_setpoint"]
        assert sp["mode"] == "enumerate"
        assert sp["legal_values"] == [0.0, 1.0, 2.0, 3.0]

    def test_translated_kwargs_actually_build_filter(self):
        # End-to-end: the translated kwargs must instantiate the rule.
        cfg = parse("@算法=l1_setpoint @参数.l1_setpoint.常态值=0 "
                    "@参数.l1_setpoint.异常值=1")
        cal = to_calibration(cfg, None)
        sp = cal.module_params["l1_setpoint"]
        f = build_filter("l1_setpoint", **sp)
        assert f.mode == "command"
        assert f.command_value == 0.0
        assert f.anomaly_values == [1.0]


class TestCalibratorParamPrecedence:
    """DSL > offline > default for module params."""

    def test_dsl_overrides_offline_param(self):
        existing = ChannelCalibration(
            module_params={"l1_sigma": {"sigma_k": 3.0}},
        )
        cfg = parse("@算法=l1_sigma @参数.l1_sigma.sigma_k=4.5")
        cal = to_calibration(cfg, existing)
        assert cal.module_params["l1_sigma"]["sigma_k"] == 4.5

    def test_offline_preserved_when_no_dsl(self):
        existing = ChannelCalibration(
            module_params={"l1_sigma": {"sigma_k": 3.0, "min_sigma": 0.1}},
        )
        cfg = parse("@算法=l1_sigma")
        cal = to_calibration(cfg, existing)
        assert cal.module_params["l1_sigma"]["sigma_k"] == 3.0
        assert cal.module_params["l1_sigma"]["min_sigma"] == 0.1

    def test_dsl_merges_with_offline_not_replaces(self):
        # DSL provides one key; offline's other key must survive.
        existing = ChannelCalibration(
            module_params={"l1_sigma": {"sigma_k": 3.0, "min_sigma": 0.1}},
        )
        cfg = parse("@算法=l1_sigma @参数.l1_sigma.sigma_k=4.0")
        cal = to_calibration(cfg, existing)
        assert cal.module_params["l1_sigma"]["sigma_k"] == 4.0  # DSL wins
        assert cal.module_params["l1_sigma"]["min_sigma"] == 0.1  # offline kept


class TestCalibratorThresholdPrecedence:
    """threshold_override from @阈值=; offline threshold preserved."""

    def test_dsl_threshold_goes_to_override(self):
        cfg = parse("@算法=tspulse @阈值=0.7")
        cal = to_calibration(cfg, None)
        assert cal.threshold_override == pytest.approx(0.7)
        # Offline threshold stays at its default (0.5) — DSL doesn't touch it.
        assert cal.threshold == 0.5

    def test_no_dsl_threshold_leaves_override_none(self):
        cfg = parse("@算法=tspulse")
        cal = to_calibration(cfg, None)
        assert cal.threshold_override is None

    def test_offline_threshold_preserved(self):
        existing = ChannelCalibration(threshold=0.42)
        cfg = parse("@算法=tspulse")
        cal = to_calibration(cfg, existing)
        assert cal.threshold == pytest.approx(0.42)
        assert cal.threshold_override is None


class TestCalibratorOfflineFieldsPreserved:
    """Offline-only fields (flip / score_type / freq_*) survive the DSL pass."""

    def test_flip_score_type_freq_preserved(self):
        existing = ChannelCalibration(
            flip=True,
            score_type="freq",
            threshold=0.55,
            threshold_name="global_p95",
            freq_band_mean=[0.1, 0.2],
            freq_band_std=[0.01, 0.02],
            freq_z_min=0.0,
            freq_z_max=1.0,
        )
        cfg = parse("@算法=tspulse @阈值=0.8")
        cal = to_calibration(cfg, existing)
        assert cal.flip is True
        assert cal.score_type == "freq"
        assert cal.threshold == 0.55
        assert cal.threshold_name == "global_p95"
        assert cal.freq_band_mean == [0.1, 0.2]
        assert cal.freq_band_std == [0.01, 0.02]
        assert cal.freq_z_min == 0.0
        assert cal.freq_z_max == 1.0
        assert cal.threshold_override == pytest.approx(0.8)


# ════════════════════════════════════════════════════════════════════════════
# Persistence round-trip
# ════════════════════════════════════════════════════════════════════════════

class TestPersistenceRoundTrip:
    """description → parse → validate → to_calibration → upsert → reload."""

    def test_round_trip_command_channel(self, tmp_path):
        cal_path = str(tmp_path / "cal.json")
        cc = CalibrationConfig(config_path=cal_path)
        desc = ("@算法=l1_setpoint @跳过模型 "
                "@参数.l1_setpoint.常态值=0 @参数.l1_setpoint.异常值=1")
        cfg = parse(desc)
        errs, _ = validate(cfg)
        assert errs == []
        cal = to_calibration(cfg, None)
        cc.upsert("T-1", cal)

        # Reload from disk and verify field equality.
        cc2 = CalibrationConfig(config_path=cal_path)
        loaded = cc2.get("T-1")
        assert loaded is not None
        assert loaded.l1_modules == ["l1_setpoint"]
        assert loaded.skip_detector is True
        assert loaded.detector_model is None
        assert loaded.module_params["l1_setpoint"]["mode"] == "command"
        assert loaded.module_params["l1_setpoint"]["command_value"] == 0.0
        assert loaded.module_params["l1_setpoint"]["anomaly_values"] == [1.0]

    def test_round_trip_three_layers(self, tmp_path):
        cal_path = str(tmp_path / "cal.json")
        cc = CalibrationConfig(config_path=cal_path)
        desc = "@算法=l1_sigma,tspulse,l3_range @阈值=0.6"
        cfg = parse(desc)
        errs, _ = validate(cfg)
        assert errs == []
        cal = to_calibration(cfg, None)
        cc.upsert("M-5", cal)

        cc2 = CalibrationConfig(config_path=cal_path)
        loaded = cc2.get("M-5")
        assert loaded is not None
        assert loaded.l1_modules == ["l1_sigma"]
        assert loaded.l3_modules == ["l3_range"]
        assert loaded.detector_model == "tspulse"
        assert loaded.threshold_override == pytest.approx(0.6)

    def test_upsert_replaces_existing_channel(self, tmp_path):
        cal_path = str(tmp_path / "cal.json")
        cc = CalibrationConfig(config_path=cal_path)
        # First save: command mode.
        cc.upsert("X-1", to_calibration(parse(
            "@算法=l1_setpoint @参数.l1_setpoint.常态值=0 "
            "@参数.l1_setpoint.异常值=1"
        ), None))
        # Second save: range mode, replaces.
        cc.upsert("X-1", to_calibration(parse(
            "@算法=l1_setpoint @参数.l1_setpoint.期望值=25 "
            "@参数.l1_setpoint.容差=2"
        ), None))
        cc2 = CalibrationConfig(config_path=cal_path)
        loaded = cc2.get("X-1")
        assert loaded.module_params["l1_setpoint"]["mode"] == "range"
        assert loaded.module_params["l1_setpoint"]["expected"] == 25.0


# ════════════════════════════════════════════════════════════════════════════
# Backward compatibility — old JSON without DSL fields
# ════════════════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    """Old channel_calibration.json (no DSL fields) must still load."""

    def test_from_dict_old_json_no_dsl_fields(self):
        old = {
            "flip": True,
            "score_type": "freq",
            "threshold": 0.42,
            "threshold_name": "global_p95",
            "freq_band_mean": [0.1],
            "freq_band_std": [0.01],
        }
        cal = ChannelCalibration.from_dict(old)
        assert cal.flip is True
        assert cal.score_type == "freq"
        assert cal.detector_model is None
        assert cal.skip_detector is False
        assert cal.threshold_override is None

    def test_from_dict_partial_dsl_fields(self):
        # JSON written by a half-deployed version: only threshold_override.
        partial = {"threshold": 0.5, "threshold_override": 0.7}
        cal = ChannelCalibration.from_dict(partial)
        assert cal.threshold == 0.5
        assert cal.threshold_override == pytest.approx(0.7)
        assert cal.detector_model is None
        assert cal.skip_detector is False

    def test_to_dict_includes_new_fields(self):
        cal = ChannelCalibration(
            detector_model="tspulse",
            skip_detector=False,
            threshold_override=0.6,
        )
        d = cal.to_dict()
        assert "detector_model" in d
        assert "skip_detector" in d
        assert "threshold_override" in d
        assert d["detector_model"] == "tspulse"
        assert d["threshold_override"] == 0.6

    def test_old_json_loads_via_calibration_config(self, tmp_path):
        cal_path = str(tmp_path / "cal.json")
        old_payload = {
            "T-1": {"flip": False, "score_type": "tsp", "threshold": 0.5},
        }
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(old_payload, f)
        cc = CalibrationConfig(config_path=cal_path)
        loaded = cc.get("T-1")
        assert loaded is not None
        assert loaded.flip is False
        assert loaded.detector_model is None


# ════════════════════════════════════════════════════════════════════════════
# The 3 canonical examples from the plan
# ════════════════════════════════════════════════════════════════════════════

class TestCanonicalExamples:
    """The three examples in the plan must parse + validate + calibrate."""

    EXAMPLE_1 = (
        "x指令通道，常态为0，异常为1 "
        "@算法=l1_setpoint @跳过模型 "
        "@参数.l1_setpoint.常态值=0 @参数.l1_setpoint.异常值=1"
    )
    EXAMPLE_2 = (
        "恒温室传感器，设定25度，波动阈值2度 "
        "@算法=l1_setpoint,tspulse "
        "@参数.l1_setpoint.期望值=25 @参数.l1_setpoint.容差=2"
    )
    EXAMPLE_3 = "电池电压，1V→5V充电5min，1min耗电 @算法=tspulse"

    def test_example_1_command_channel(self):
        cfg = parse(self.EXAMPLE_1)
        errs, warns = validate(cfg)
        assert errs == []
        cal = to_calibration(cfg, None)
        assert cal.l1_modules == ["l1_setpoint"]
        assert cal.skip_detector is True
        assert cal.detector_model is None
        # The setpoint params must build a working filter.
        f = build_filter("l1_setpoint", **cal.module_params["l1_setpoint"])
        assert f.mode == "command"

    def test_example_2_thermostat(self):
        cfg = parse(self.EXAMPLE_2)
        errs, warns = validate(cfg)
        assert errs == []
        cal = to_calibration(cfg, None)
        assert cal.l1_modules == ["l1_setpoint"]
        assert cal.detector_model == "tspulse"
        f = build_filter("l1_setpoint", **cal.module_params["l1_setpoint"])
        assert f.mode == "range"
        assert f.expected == 25.0

    def test_example_3_battery_voltage(self):
        cfg = parse(self.EXAMPLE_3)
        errs, warns = validate(cfg)
        assert errs == []
        cal = to_calibration(cfg, None)
        # L1/L3 fall back to None (system default); L2 = tspulse.
        assert cal.l1_modules is None
        assert cal.l3_modules is None
        assert cal.detector_model == "tspulse"


# ════════════════════════════════════════════════════════════════════════════
# Registry sanity — the validator's name-legality source
# ════════════════════════════════════════════════════════════════════════════

class TestRegistryAssumptions:
    """Guard against silent drift in FILTER_REGISTRY / MODEL_REGISTRY.

    These tests exist because the validator's E1 dynamically consults the
    registries; if a name we depend on in the examples ever gets renamed,
    we want a clear failure here rather than a mysterious E1 in the field.
    """

    def test_l1_setpoint_registered(self):
        assert "l1_setpoint" in FILTER_REGISTRY

    def test_l1_sigma_registered(self):
        assert "l1_sigma" in FILTER_REGISTRY

    def test_l3_range_registered(self):
        assert "l3_range" in FILTER_REGISTRY

    def test_tspulse_is_detector(self):
        assert "tspulse" in MODEL_REGISTRY
        assert MODEL_REGISTRY["tspulse"].kind == "detector"

    def test_default_l1_modules_stable(self):
        assert DEFAULT_L1_MODULES == [
            "l1_constant", "l1_sigma", "l1_iqr", "l1_rate",
        ]

    def test_default_l3_modules_stable(self):
        assert DEFAULT_L3_MODULES == [
            "l3_nan_sanitise", "l3_constant", "l3_range",
            "l3_rate", "l3_variance",
        ]

    def test_commands_catalogue_complete(self):
        # The four canonical commands must all be present.
        assert set(COMMANDS.keys()) == {"算法", "跳过模型", "阈值", "参数"}
