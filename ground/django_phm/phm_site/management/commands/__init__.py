"""Agent-friendly CLI commands for the PHM system.

Each command here mirrors a user-facing operation so that automated agents
(and operators) can drive the system without a browser.  Commands share the
same service layer as the HTTP API — no duplicated business logic.

Conventions:
  * Every command accepts ``--format json`` for machine-readable output
    (default is human-readable text).
  * Read-only commands (status / list) are safe to run anytime.
  * Write commands require explicit confirmation flags where destructive
    (only ``device save`` and ``device rm`` need ``--confirm`` — they
    overwrite ``device_config.json``; alert verdict/status and diagnosis
    triggers are reversible/idempotent and need no confirmation).

Available commands:
  * ``manage.py rul``       — RUL service status, source list, predictions.
  * ``manage.py models``    — registered model registry (detector/forecaster/RUL).
  * ``manage.py config``    — system runtime configuration dump.
  * ``manage.py device``    — device tree show / save / rm.
  * ``manage.py alert``     — alert list / verdict / status.
  * ``manage.py diagnose``  — LLM diagnosis run / auto / status / list.
  * ``manage.py export``    — telemetry export (csv / json).

Shared output helpers live in ``_common.py`` (emit / ok_payload /
error_payload / not_found_payload).  The leading underscore keeps Django
from treating it as a command module.
"""
