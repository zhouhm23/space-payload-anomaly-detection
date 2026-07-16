"""Agent-friendly CLI commands for the PHM system.

Each command here mirrors a user-facing operation so that automated agents
(and operators) can drive the system without a browser.  Commands share the
same service layer as the HTTP API — no duplicated business logic.

Conventions:
  * Every command accepts ``--format json`` for machine-readable output
    (default is human-readable text).
  * Read-only commands (status / list) are safe to run anytime.
  * Write commands require explicit confirmation flags where destructive.

Available commands:
  * ``manage.py rul`` — RUL service status, source list, predictions.
"""
