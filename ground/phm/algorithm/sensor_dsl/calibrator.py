"""Map a validated :class:`SensorConfig` onto a :class:`ChannelCalibration`.

The calibrator is the persistence adapter between the DSL world (Chinese
parameter names, layered by ``@算法=``) and the runtime world
(:class:`ChannelCalibration` fields consumed by the cascade).  It applies
the user-prioritised precedence chain:

    DSL explicit (@参数)  >  offline calibration JSON  >  algorithm default

Operationally:
  * ``cfg.algorithms`` is split by layer prefix into ``l1_modules`` /
    ``l3_modules``; L2 detector models go to ``detector_model``.
  * ``cfg.params[module]`` is translated to Python constructor kwargs.
    For ``l1_setpoint`` this includes choosing the rule ``mode`` from the
    anchor key(s) present (command / range / enumerate).  For other
    modules the DSL key is the kwarg name verbatim.
  * ``cfg.threshold`` (from ``@阈值=``) populates ``threshold_override``;
    the offline ``threshold`` field is left untouched so the cascade can
    fall back to it when ``threshold_override`` is None.
  * ``existing`` offline calibration (direction flip, score type, freq
    baseline, offline threshold) is preserved — DSL never overwrites
    data-derived fields, only flow-derived ones.

This function does not validate; it assumes the caller already ran
:func:`sensor_dsl.validator.validate` and got an empty errors list.
"""

from __future__ import annotations

from typing import Any

from ..calibration_config import ChannelCalibration
from ..rules import DEFAULT_L1_MODULES, DEFAULT_L3_MODULES
from .commands import SensorConfig
from .validator import classify_layer, DEFAULT_DETECTOR_MODEL


__all__ = ["to_calibration"]


# Chinese DSL parameter key → l1_setpoint constructor kwarg.
# Used to translate the user-facing names into the Python-facing names that
# ``build_filter("l1_setpoint", **kwargs)`` expects.  The mode is chosen
# from which anchor group is present (see ``_resolve_setpoint_mode``).
_SETPOINT_KEY_MAP = {
    "常态值": "command_value",     # command mode
    "异常值": "anomaly_values",    # command mode (comma-separated list)
    "期望值": "expected",         # range mode
    "容差": "tolerance",         # range mode
    "范围下限": "range_low",      # range mode (explicit band)
    "范围上限": "range_high",     # range mode (explicit band)
    "合法值": "legal_values",     # enumerate mode (comma-separated list)
}


def _to_float_list(raw: Any) -> list[float]:
    """Coerce a comma-separated string or a list into a list of floats.

    ``@参数.l1_setpoint.异常值=1,2`` parses to the raw string ``"1,2"``;
    the validator confirms it is non-empty, but the calibrator must still
    turn it into ``[1.0, 2.0]`` for the rule constructor.  Items that
    don't parse as float are silently dropped — by the time we get here,
    the validator has already accepted the config, and a scientist writing
    a deliberately bad list value will see it surface at build_filter time.
    """
    if isinstance(raw, list):
        out: list[float] = []
        for x in raw:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        return out
    if not isinstance(raw, str):
        # A single coerced float (parser already turned ``"1"`` into 1.0).
        try:
            return [float(raw)]
        except (TypeError, ValueError):
            return []
    pieces = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for p in pieces:
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def _to_float(raw: Any) -> float | None:
    """Coerce to float, returning None on failure."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _resolve_setpoint_mode(
    dsl_params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Pick the l1_setpoint ``mode`` and translate DSL keys to kwargs.

    Returns ``(mode, constructor_kwargs)``.  Mode selection mirrors the
    rule's own parameter validation (see ``L1SetpointRule._validate_args``):

      * If ``合法值`` present  → enumerate mode.
      * Elif ``常态值`` or ``异常值`` present → command mode.
      * Elif ``期望值``/``容差`` or ``范围下限``/``范围上限`` → range mode.
      * Else → range mode with empty kwargs (the validator's E3 would
        already have blocked this; we default defensively to avoid a
        crash on direct programmatic use).

    Only keys that map to the chosen mode are emitted; keys belonging to
    other modes are dropped so the rule constructor doesn't see conflicting
    arguments (e.g. a stray ``合法值`` while in command mode).
    """
    has_enumerate = "合法值" in dsl_params
    has_command = "常态值" in dsl_params or "异常值" in dsl_params
    has_range_pair = (
        ("期望值" in dsl_params or "范围下限" in dsl_params)
    )

    if has_enumerate:
        mode = "enumerate"
        kw: dict[str, Any] = {}
        if "合法值" in dsl_params:
            vals = _to_float_list(dsl_params["合法值"])
            if vals:
                kw["legal_values"] = vals
        return mode, kw

    if has_command:
        mode = "command"
        kw = {}
        if "常态值" in dsl_params:
            v = _to_float(dsl_params["常态值"])
            if v is not None:
                kw["command_value"] = v
        if "异常值" in dsl_params:
            vals = _to_float_list(dsl_params["异常值"])
            if vals:
                kw["anomaly_values"] = vals
        return mode, kw

    if has_range_pair:
        mode = "range"
        kw = {}
        if "期望值" in dsl_params:
            v = _to_float(dsl_params["期望值"])
            if v is not None:
                kw["expected"] = v
        if "容差" in dsl_params:
            v = _to_float(dsl_params["容差"])
            if v is not None:
                kw["tolerance"] = v
        if "范围下限" in dsl_params:
            v = _to_float(dsl_params["范围下限"])
            if v is not None:
                kw["range_low"] = v
        if "范围上限" in dsl_params:
            v = _to_float(dsl_params["范围上限"])
            if v is not None:
                kw["range_high"] = v
        return mode, kw

    # Defensive default — validator should have caught this (E3).
    return "range", {}


def _translate_module_params(
    module_name: str,
    dsl_params: dict[str, Any],
) -> dict[str, Any]:
    """Translate one module's DSL params to constructor kwargs.

    For ``l1_setpoint``: applies the Chinese→kwarg map and chooses mode.
    For every other module: assumes the DSL key **is** the kwarg name
    verbatim (e.g. ``@参数.l1_sigma.sigma_k=4`` → ``{"sigma_k": 4.0}``).
    This keeps the calibrator forward-compatible with future rule modules
    without a per-module translation table; the only special case is
    setpoint because its user-facing vocabulary is deliberately Chinese.
    """
    if module_name == "l1_setpoint":
        mode, kw = _resolve_setpoint_mode(dsl_params)
        # Don't write an empty kwargs dict — leave the mode decision to
        # the constructor default if nothing resolved (defensive).
        out: dict[str, Any] = {}
        out["mode"] = mode
        out.update(kw)
        return out

    out = {}
    for key, val in dsl_params.items():
        # Scalars: parser already coerced numeric strings to float.
        # Lists: leave as-is so a future module accepting list kwargs works.
        out[key] = val
    return out


def to_calibration(
    cfg: SensorConfig,
    existing: ChannelCalibration | None,
) -> ChannelCalibration:
    """Produce the :class:`ChannelCalibration` to persist for this sensor.

    Applies explicit-over-default semantics:

      * ``l1_modules``: from cfg's ``l1_*`` names, else ``existing.l1_modules``,
        else ``None`` (which the runtime treats as DEFAULT_L1_MODULES).
      * ``l3_modules``: same logic for ``l3_*``.
      * ``detector_model``: the L2 model in ``@算法=``, else None.
      * ``skip_detector``: direct passthrough.
      * ``module_params``: translated kwargs, DSL wins over existing.
      * ``threshold_override``: from ``@阈值=``, else None.
      * offline-only fields (``flip`` / ``score_type`` / ``threshold`` /
        ``freq_*``): copied verbatim from ``existing``.

    Args:
        cfg: parsed and validated SensorConfig.
        existing: the channel's current ChannelCalibration (offline
            calibration), or None when the channel has no prior record.

    Returns:
        A new :class:`ChannelCalibration` ready to ``upsert``.
    """
    ex = existing or ChannelCalibration()

    # ── Layer membership from algorithm names ─────────────────────────
    l1_modules: list[str] | None = None
    l3_modules: list[str] | None = None
    detector_model: str | None = None
    for name in cfg.algorithms:
        layer = classify_layer(name)
        if layer == "L1":
            if l1_modules is None:
                l1_modules = []
            l1_modules.append(name)
        elif layer == "L3":
            if l3_modules is None:
                l3_modules = []
            l3_modules.append(name)
        elif layer == "L2":
            detector_model = name
        # Non-layer models (forecaster / rul) are accepted by the validator
        # but do not influence the cascade chain — left out of module lists.

    # If @算法= didn't list any L1/L3 names, fall back to the existing
    # calibration's module lists (so an offline recommendation isn't wiped
    # just because the scientist only wrote @算法=tspulse).
    if l1_modules is None:
        l1_modules = list(ex.l1_modules) if ex.l1_modules else None
    if l3_modules is None:
        l3_modules = list(ex.l3_modules) if ex.l3_modules else None

    # ── Module parameter overrides (DSL > offline) ────────────────────
    merged_params: dict[str, dict[str, Any]] = {}
    if ex.module_params:
        for m, kvs in ex.module_params.items():
            merged_params[m] = dict(kvs)
    for module_name, dsl_params in cfg.params.items():
        translated = _translate_module_params(module_name, dsl_params)
        bucket = merged_params.setdefault(module_name, {})
        # DSL wins on key conflicts.
        bucket.update(translated)

    # ── Detector / threshold fields ───────────────────────────────────
    # When the scientist wrote @跳过模型, skip_detector is authoritative
    # and detector_model must be None regardless of what @算法= said
    # (the validator's E2 ensures they aren't both set, but be defensive).
    skip_detector = cfg.skip_detector
    if skip_detector:
        detector_model = None
    elif detector_model is None and not cfg.is_empty and cfg.algorithms:
        # Scientist listed algorithms but no L2 detector → keep None so
        # the runtime knows to use the system default detector.
        detector_model = None

    threshold_override = None
    if cfg.threshold is not None and isinstance(cfg.threshold, (int, float)):
        thr = float(cfg.threshold)
        if 0.0 <= thr <= 1.0:
            threshold_override = thr

    return ChannelCalibration(
        # Offline-only fields: preserved verbatim from existing.
        flip=ex.flip,
        score_type=ex.score_type,
        threshold=ex.threshold,
        threshold_name=ex.threshold_name,
        freq_band_mean=list(ex.freq_band_mean) if ex.freq_band_mean else None,
        freq_band_std=list(ex.freq_band_std) if ex.freq_band_std else None,
        freq_z_min=ex.freq_z_min,
        freq_z_max=ex.freq_z_max,
        # DSL-derived flow fields.
        l1_modules=l1_modules,
        l3_modules=l3_modules,
        module_params=merged_params if merged_params else None,
        detector_model=detector_model,
        skip_detector=skip_detector,
        threshold_override=threshold_override,
    )
