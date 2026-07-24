"""Validate a parsed :class:`SensorConfig`. Returns ``(errors, warnings)``.

The validator replaces the v1 "automatic algorithm router".  Per the four
iron rules of the v2 plan:

  * It only checks the config is **legal and complete** (a valid pipeline
    can be assembled).  It never picks "the best" algorithm — that needs
    labels and lives in the offline recommender.
  * Hard errors (E1-E5) **block the device-tree save** with HTTP 400.
    Warnings (W1-W2) pass through but surface in the editor so the
    scientist knows which layers fall back to system default.

Layer inference (rule 4 of the plan: "算法名即层归属"):
  * ``l1_*`` names → L1 rule chain.
  * ``l3_*`` names → L3 physical post-processing.
  * :data:`MODEL_REGISTRY` entries with ``kind == "detector"`` → L2 model
    (currently only ``tspulse``; the kind check is dynamic so a new
    detector added later is picked up without touching this file).
  * Other model kinds (forecaster / rul) are accepted as algorithm names
    but do not belong to a cascade layer — they are out-of-band concerns
    (the ``@rul`` channel specialisation is handled elsewhere).

Algorithm-name legitimacy (E1) is checked dynamically against
:data:`FILTER_REGISTRY` ∪ :data:`MODEL_REGISTRY` so this module never goes
stale when a new rule or model is registered.
"""

from __future__ import annotations

from typing import Any

from .._registry import MODEL_REGISTRY
from ..rules import FILTER_REGISTRY, DEFAULT_L1_MODULES, DEFAULT_L3_MODULES
from .commands import SensorConfig


__all__ = ["validate", "classify_layer", "DEFAULT_DETECTOR_MODEL"]


# Canonical default L2 detector (referenced in W1 messages and by the
# calibrator).  ``tspulse`` is the only kind="detector" entry today; if a
# future registry swap changes this, scan MODEL_REGISTRY for the first
# detector instead of hard-coding.
DEFAULT_DETECTOR_MODEL = "tspulse"


# l1_setpoint "anchor" parameter keys.  At least one must be present when
# the scientist writes ``@算法=l1_setpoint`` — without one of these the
# rule has no physical expectation to check against and cannot be built
# (its constructor raises).  Keys are the Chinese DSL names (the parser
# stores whatever the scientist wrote, so we match on the user-facing name
# space here, not the Python constructor kwargs).
_SETPOINT_ANCHOR_KEYS = {
    "常态值",   # command mode: command_value
    "异常值",   # command mode: anomaly_values
    "期望值",   # range mode: expected
    "容差",     # range mode: tolerance
    "范围下限",  # range mode: range_low
    "范围上限",  # range mode: range_high
    "合法值",   # enumerate mode: legal_values
}


def classify_layer(name: str) -> str | None:
    """Return the cascade layer a name belongs to, or ``None`` if ambiguous.

    Used by E2 (L2 vs @跳过模型 mutex) and W1 (which layers are covered).
    Names are classified by prefix convention per the plan's rule 4:

      * ``l1_*`` → ``"L1"``
      * ``l3_*`` → ``"L3"``
      * MODEL_REGISTRY entry with ``kind == "detector"`` → ``"L2"``
      * anything else (forecaster / rul / unknown) → ``None``

    A name not in FILTER_REGISTRY ∪ MODEL_REGISTRY is still classified by
    prefix so E1 can produce a layer-aware error message, but E1 itself
    will already flag it as illegal.
    """
    if name.startswith("l1_"):
        return "L1"
    if name.startswith("l3_"):
        return "L3"
    entry = MODEL_REGISTRY.get(name)
    if entry is not None and entry.kind == "detector":
        return "L2"
    return None


def _is_known_algorithm(name: str) -> bool:
    """True iff ``name`` is in FILTER_REGISTRY or MODEL_REGISTRY."""
    return name in FILTER_REGISTRY or name in MODEL_REGISTRY


def validate(cfg: SensorConfig) -> tuple[list[str], list[str]]:
    """Validate a parsed :class:`SensorConfig`.

    Args:
        cfg: the parser output for one sensor description.

    Returns:
        ``(errors, warnings)``.  ``errors`` is non-empty iff the device-tree
        save should be blocked (HTTP 400).  ``warnings`` are advisory.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # W2: completely empty description → full system default flow.  This
    # is legal (defaults are explicitly defined) but worth surfacing so the
    # scientist knows the channel is running with zero customisation.
    if cfg.is_empty:
        warnings.append(
            "未检测到任何 @ 命令，将走系统默认全流程"
            f"（L1={DEFAULT_L1_MODULES}, L2={DEFAULT_DETECTOR_MODEL}, "
            f"L3={DEFAULT_L3_MODULES}）"
        )
        return errors, warnings

    # ── E1: every algorithm name must be a known filter or model ──────
    for name in cfg.algorithms:
        if not _is_known_algorithm(name):
            errors.append(
                f"E1: 算法名 {name!r} 不在 FILTER_REGISTRY ∪ MODEL_REGISTRY 中"
                "（可用名见算法库页或 rules/__init__.py）"
            )

    # ── E2: @跳过模型 cannot coexist with an L2 detector in @算法= ────
    if cfg.skip_detector:
        l2_models = [
            n for n in cfg.algorithms if classify_layer(n) == "L2"
        ]
        if l2_models:
            errors.append(
                "E2: @跳过模型 与 @算法= 中的 L2 检测器互斥"
                f"（@算法= 含 {l2_models}，但又写了 @跳过模型）"
            )

    # ── E3: l1_setpoint needs at least one anchor parameter ───────────
    if "l1_setpoint" in cfg.algorithms:
        setpoint_params = cfg.params.get("l1_setpoint", {})
        present_anchors = _SETPOINT_ANCHOR_KEYS & set(setpoint_params.keys())
        if not present_anchors:
            errors.append(
                "E3: @算法=l1_setpoint 必须至少配置一个锚点参数"
                f"（{_SETPOINT_ANCHOR_KEYS}），"
                "否则规则无法构建（三种模式各自所需的物理期望缺失）"
            )

    # ── E4: @阈值 must be a number in [0, 1] ──────────────────────────
    if cfg.threshold is not None:
        if not isinstance(cfg.threshold, (int, float)):
            errors.append(
                f"E4: @阈值= 必须是 [0,1] 内的数字，got {cfg.threshold!r}"
            )
        elif not (0.0 <= float(cfg.threshold) <= 1.0):
            errors.append(
                f"E4: @阈值= 必须在 [0,1] 范围内，got {cfg.threshold}"
            )

    # ── E5: @参数.<module>.* may only target modules in @算法= ────────
    declared = set(cfg.algorithms)
    for module_name in cfg.params.keys():
        # l1_setpoint is allowed to be parametrised even when listed in
        # @算法= (E3 already enforces that).  Generic rule: the parametrised
        # module must appear in the declared algorithm list.
        if module_name not in declared:
            errors.append(
                f"E5: @参数.{module_name}.* 配置了参数，"
                f"但 {module_name!r} 不在 @算法= 列表里"
                "（不能为未启用的模块配参数）"
            )

    # ── W1: layers not covered by @算法= fall back to system default ──
    # Only emit when the scientist wrote *something* in @算法= (otherwise
    # W2 above already covered the "fully default" case).
    if cfg.algorithms:
        covered_layers = {classify_layer(n) for n in cfg.algorithms}
        # L1 default applies when no l1_* name was declared AND skip_detector
        # doesn't change L1 defaulting (skip is an L2 concern).
        if "L1" not in covered_layers:
            warnings.append(
                f"W1: @算法= 未覆盖 L1 层，将走默认 L1={DEFAULT_L1_MODULES}"
            )
        # L2: skipped explicitly, or not declared.
        if cfg.skip_detector:
            warnings.append("W1: @跳过模型 已声明，L2 层将被跳过")
        elif "L2" not in covered_layers:
            warnings.append(
                f"W1: @算法= 未指定 L2 检测器，将走默认 L2={DEFAULT_DETECTOR_MODEL}"
            )
        if "L3" not in covered_layers:
            warnings.append(
                f"W1: @算法= 未覆盖 L3 层，将走默认 L3={DEFAULT_L3_MODULES}"
            )

    return errors, warnings
