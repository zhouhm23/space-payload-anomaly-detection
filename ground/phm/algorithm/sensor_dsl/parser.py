"""Parse a sensor description into a raw :class:`SensorConfig`. Never raises.

Tokenises the ``@<name>=<value>`` / ``@<flag>`` commands out of free text
but performs **no semantic validation** — that is the validator's job.
Unknown ``@tokens`` (e.g. ``@备注=foo`` or the legacy ``@rul:fd001``) are
silently treated as prose and do not appear in the output.  This keeps the
DSL forgiving: a scientist mixing free-form notes with commands never sees
a parse error.

Grammar (informal)::

    description := <free text with embedded @commands>
    @command    := "@" name ["=" value]
    name        := run of chars other than whitespace, "@" and "="
    value       := run of chars other than whitespace and "@"

Notes on the regex:
  * The value charset excludes ``@`` (so two adjacent tokens like
    ``@参数.x.a=0 @参数.y.b=1`` parse as two tokens, not one big value).
  * The name charset also excludes ``=``; a name with embedded ``=`` would
    be ambiguous and is treated as prose.
  * Chinese punctuation (full-width space ``　``, commas ``，``) is **not**
    in the value charset — scientists are expected to use ASCII ``,`` to
    separate algorithm names in ``@算法=`` (the DSL manual documents this
    explicitly).  A full-width comma would terminate the value early.
"""

from __future__ import annotations

import re

from .commands import COMMANDS, SensorConfig


__all__ = ["parse"]


# Captures ``@<name>`` optionally followed by ``=<value>``.  The value stops
# at the next whitespace OR the next ``@`` (so adjacent tokens don't merge).
_TOKEN_RE = re.compile(r"@([^\s=@]+)(?:=([^\s@]+))?")


def _coerce_float(raw: str) -> float | str:
    """Best-effort float coercion.  Returns the raw string on failure.

    The parser never raises — a malformed numeric value is passed through
    as a string so the validator can produce a precise error message
    (``@阈值=abc`` → ``threshold`` holds ``"abc"``, validator's E4 fires).
    """
    try:
        return float(raw)
    except (TypeError, ValueError):
        return raw


def _assign(cfg: SensorConfig, head: str, value: str | None) -> None:
    """Route a recognised token to the right :class:`SensorConfig` field.

    ``head`` is the canonical command head (``算法`` / ``跳过模型`` /
    ``阈值`` / ``参数``).  For ``参数`` the full parsed name is
    ``参数.<module>.<key>`` — the caller splits off the path suffix before
    dispatching so this function only sees the head.
    """
    if head == "算法":
        # Split on ASCII comma; drop empties from trailing/double commas.
        if value is None:
            return
        for piece in value.split(","):
            piece = piece.strip()
            if piece:
                cfg.algorithms.append(piece)
    elif head == "跳过模型":
        cfg.skip_detector = True
    elif head == "阈值":
        if value is None:
            return
        cfg.threshold = _coerce_float(value)
    elif head == "参数":
        # value has already been resolved to the key path by the caller;
        # we receive (module, key, value) via the dedicated path below.
        raise RuntimeError("参数 must be dispatched via the param-path branch")


def _assign_param(cfg: SensorConfig, full_name: str, value: str | None) -> None:
    """Handle the ``@参数.<module>.<key>=<value>`` form.

    The parser is permissive here — it stores whatever module/key the
    scientist wrote, even if the module name is unknown.  The validator's
    E5 checks whether the module actually appears in ``@算法=``.
    """
    if value is None:
        return  # ``@参数.x.y`` with no ``=`` — silently ignore.
    # Strip the leading "参数." prefix; what remains is ``<module>.<key>``.
    suffix = full_name[len("参数") + 1:]  # +1 for the "."
    if "." not in suffix:
        # ``@参数.module=...`` with no key — malformed, leave to validator.
        module, key = suffix, ""
    else:
        module, key = suffix.split(".", 1)
    if not module:
        return
    cfg.params.setdefault(module, {})[key] = _coerce_float(value)


def parse(description: str | None) -> SensorConfig:
    """Parse a sensor description into a :class:`SensorConfig`.

    Never raises.  ``None`` / empty / whitespace-only input returns an
    empty :class:`SensorConfig` (which the validator will flag as W2:
    "no commands → system default flow").

    Args:
        description: the free-text sensor description, possibly containing
            embedded ``@commands``.

    Returns:
        A populated :class:`SensorConfig`.  Unknown ``@tokens`` do not
        appear in the result.
    """
    cfg = SensorConfig()
    if not description:
        return cfg
    for m in _TOKEN_RE.finditer(description):
        name = m.group(1)
        value = m.group(2)
        cfg.raw_tokens.append((name, value))

        # ``参数`` is special: the full token name is ``参数.<module>.<key>``
        # (a dotted path), so we cannot just look it up in COMMANDS.
        if name == "参数" or name.startswith("参数."):
            _assign_param(cfg, name, value)
            continue

        spec = COMMANDS.get(name)
        if spec is None:
            continue  # unknown @token → prose
        _assign(cfg, name, value)
    return cfg
