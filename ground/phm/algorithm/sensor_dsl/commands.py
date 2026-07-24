"""@command specification — canonical names, value types, raw config shape.

This module is the home of two pure-data artefacts consumed by the parser,
validator and calibrator:

  * :data:`COMMANDS` — the canonical command-name → spec mapping.  The parser
    looks tokens up here; anything not in this dict is treated as prose and
    silently ignored (the DSL never raises on unknown tokens).
  * :class:`SensorConfig` — the raw parsed configuration produced by
    :func:`sensor_dsl.parser.parse`.  It is deliberately a dumb struct: the
    parser populates it, the validator checks it, the calibrator maps it
    onto :class:`ChannelCalibration`.  Keeping it here (rather than in
    ``parser.py``) breaks what would otherwise be a circular import:
    ``validator`` and ``calibrator`` both need the type.

Algorithm-name legality is **not** encoded here.  The validator looks the
names up dynamically in :data:`FILTER_REGISTRY` and :data:`MODEL_REGISTRY`
so this spec does not go stale when a new rule or model is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


__all__ = ["CommandSpec", "COMMANDS", "SensorConfig"]


@dataclass(frozen=True)
class CommandSpec:
    """Declarative spec for one @command.

    Attributes:
        name: canonical command name (e.g. ``"算法"``).
        value_type: one of ``"list"`` / ``"float"`` / ``"flag"`` /
            ``"param_path"``.  The parser uses this to type-coerce the
            raw token value.
        description: human-readable summary (used in error messages and
            in the syntax manual).
    """

    name: str
    value_type: str
    description: str


# Canonical command catalogue.  The parser matches the literal token after
# ``@`` against :data:`COMMANDS` keys; anything else is prose.  Algorithm-name
# legitimacy is checked in the validator against FILTER_REGISTRY ∪ MODEL_REGISTRY.
COMMANDS: dict[str, CommandSpec] = {
    "算法": CommandSpec(
        name="算法",
        value_type="list",
        description="处理流算法/模型名列表（逗号分隔，名字来自 FILTER_REGISTRY / MODEL_REGISTRY）",
    ),
    "跳过模型": CommandSpec(
        name="跳过模型",
        value_type="flag",
        description="显式跳过 L2 模型（用于指令/状态通道）",
    ),
    "阈值": CommandSpec(
        name="阈值",
        value_type="float",
        description="全局异常分数触发线，无量纲 [0,1]",
    ),
    "参数": CommandSpec(
        name="参数",
        value_type="param_path",
        description="模块参数覆盖，格式 @参数.<模块>.<键>=<值>",
    ),
}


@dataclass
class SensorConfig:
    """Raw parsed @command configuration for one sensor description.

    This is the parser's output: it carries the literal tokens the
    scientist wrote, with only light type coercion (lists split on ``,``
    and floats parsed).  Semantic checking happens in
    :func:`sensor_dsl.validator.validate`; persistence mapping happens in
    :func:`sensor_dsl.calibrator.to_calibration`.

    Attributes:
        algorithms: list of algorithm/model names from ``@算法=`` (in the
            order written).  Empty when the scientist did not write
            ``@算法=``.
        skip_detector: True iff ``@跳过模型`` appeared.
        threshold: the float from ``@阈值=`` or ``None`` when absent.
        params: nested dict ``{module_name: {key: float_value}}`` from
            ``@参数.<module>.<key>=<value>``.  Values are coerced to float
            when they parse, otherwise kept as the raw string.
        raw_tokens: list of ``(name, value)`` tuples in source order — kept
            for diagnostics / error messages; not used by the validator's
            decision logic.
    """

    algorithms: list[str] = field(default_factory=list)
    skip_detector: bool = False
    threshold: float | None = None
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_tokens: list[tuple[str, str | None]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when no @command was recognised at all (full default flow)."""
        return (
            not self.algorithms
            and not self.skip_detector
            and self.threshold is None
            and not self.params
        )
