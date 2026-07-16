"""Shared helpers for agent-friendly CLI commands.

Centralises the JSON/text output pattern duplicated across the original
three commands (rul / models / config).  New commands (device / alert /
diagnose / export) import from here so payload construction and stdout
formatting stay consistent.

Design rules (see AGENTS.md "Agent 友好双通道"):
  * Every command supports ``--format json`` for machine-readable output.
  * JSON payloads always carry a ``status`` field (ok / disabled /
    not_found / error) so agents can branch on it without parsing text.
  * ``json.dumps`` uses ``ensure_ascii=False`` (Chinese messages) and
    ``indent=2`` (human-friendly when piped to a file).
"""

from __future__ import annotations

import json

# Output format choices shared by every command.  Importing this constant
# keeps the ``add_argument`` calls identical across commands.
FORMAT_CHOICES = ("text", "json")
FORMAT_HELP = "Output format (default: text; json for agents)."


def emit(out, payload: dict, fmt: str) -> None:
    """Write ``payload`` in the requested format to ``out``.

    ``out`` is a Django ``Command.stdout`` (or ``stderr``) stream.  When
    ``fmt == "json"`` the payload is serialised with ``ensure_ascii=False``
    and ``indent=2``.  Otherwise the caller is responsible for rendering
    text — pass ``render=None`` and call :func:`emit_text` separately, or
    pass a ``render`` callable that takes ``(out, payload)``.

    This helper only handles the JSON branch so callers can compose their
    own text rendering without re-implementing the JSON dump.
    """
    if fmt == "json":
        out.write(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    # Text mode is intentionally a no-op here — callers render via their
    # own ``_render_text`` methods.  This keeps the JSON path testable in
    # isolation and avoids forcing a shared text layout on commands whose
    # output shapes differ widely (tables vs trees vs markdown).
    raise ValueError(
        "emit() handles json only; for text mode call the command's "
        "_render_text directly."
    )


def ok_payload(**data) -> dict:
    """Build a success body: ``{"status": "ok", **data}``."""
    return {"status": "ok", **data}


def error_payload(message: str, **extra) -> dict:
    """Build a structured error body: ``{"status": "error", "message": ...}``.

    Extra fields (e.g. ``channel``, ``node_id``) help agents pinpoint the
    failing resource without re-running the command.
    """
    return {"status": "error", "message": message, **extra}


def not_found_payload(resource: str, ident, **extra) -> dict:
    """Build a not-found body: ``{"status": "not_found", "<resource>": ident}``.

    ``resource`` is the field name (e.g. ``"channel"``, ``"node_id"``);
    ``ident`` is the value the caller asked for.
    """
    return {"status": "not_found", resource: ident, **extra}


__all__ = [
    "FORMAT_CHOICES",
    "FORMAT_HELP",
    "emit",
    "ok_payload",
    "error_payload",
    "not_found_payload",
]
