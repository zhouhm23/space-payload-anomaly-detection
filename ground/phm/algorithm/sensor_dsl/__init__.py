"""@command DSL — parse, validate and persist sensor pipeline specs.

This package turns the free-text ``description`` field of a device-tree
sensor node into a validated, persisted per-channel pipeline.  It is the
v2 replacement for the (abandoned) automatic algorithm router: the
scientist explicitly writes the desired processing flow using Chinese
``@commands``, the validator blocks illegal configs at save time, and the
calibrator maps the result onto a :class:`ChannelCalibration` that the
runtime cascade consumes.

Public surface (three callables + the data types they exchange)::

    parse(description)        -> SensorConfig      # parser, never raises
    validate(cfg)             -> (errors, warnings)  # E1-E5 / W1-W2
    to_calibration(cfg, ex)   -> ChannelCalibration  # persistence adapter

Typical call sequence (as wired by ``device_tree_save_api``)::

    cfg = parse(node["description"])
    errors, warnings = validate(cfg)
    if errors:
        return JsonResponse({...}, status=400)   # hard block
    cal = to_calibration(cfg, existing_for_channel)
    CalibrationConfig().upsert(channel, cal)

See ``docs/product/@命令语法说明书.md`` for the user-facing syntax manual.
"""

from __future__ import annotations

from .calibrator import to_calibration
from .commands import CommandSpec, COMMANDS, SensorConfig
from .parser import parse
from .validator import classify_layer, validate


__all__ = [
    "parse",
    "validate",
    "to_calibration",
    "classify_layer",
    "SensorConfig",
    "CommandSpec",
    "COMMANDS",
]
