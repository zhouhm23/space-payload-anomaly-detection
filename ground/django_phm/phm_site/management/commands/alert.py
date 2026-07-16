"""Agent-friendly CLI for alerts (list / verdict / status).

Mirrors three HTTP endpoints:
  * ``GET /api/alerts``            — in-memory live alerts (``--live``)
  * ``GET /api/alerts/history``    — persisted DB alerts (default)
  * ``POST /api/alerts/verdict``   — annotate human verdict
  * ``PATCH /api/alerts/<id>``     — update lifecycle status

Verdict and status are single-row annotations — fully reversible (re-run
the command with a different value).  Therefore **no ``--confirm`` flag**
is required, unlike ``device save/rm``.

Usage::

    # List (default: DB history; --live switches to in-memory current alerts)
    manage.py alert list
    manage.py alert list --limit 20
    manage.py alert list --live
    manage.py alert list --format json

    # Annotate a verdict (real / false_alarm / uncertain)
    manage.py alert verdict <channel> <alert_ts> <verdict>
    manage.py alert verdict C-1 1752633600.0 real
    manage.py alert verdict C-1 1752633600.0 false_alarm --format json

    # Update lifecycle status (pending / confirmed / false)
    manage.py alert status <alert_id> <status>
    manage.py alert status 12 confirmed
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from phm_site.services_bridge import get_container

from ._common import (
    FORMAT_CHOICES,
    FORMAT_HELP,
    emit,
    error_payload,
    not_found_payload,
    ok_payload,
)

# Echo the service-layer constants so the CLI rejects bad values with a
# clear message rather than letting SQLiteStore.update_* silently return False.
_VALID_VERDICTS = ("real", "false_alarm", "uncertain")
_VALID_STATUSES = ("pending", "confirmed", "false")


class Command(BaseCommand):
    help = "Query or annotate alerts (list / verdict / status)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "action",
            choices=["list", "verdict", "status"],
            help="list: recent alerts. "
                 "verdict: annotate real/false_alarm/uncertain. "
                 "status: set pending/confirmed/false.",
        )
        parser.add_argument(
            "--limit", type=int, default=50,
            help="For 'list': max alerts to show (default 50).",
        )
        parser.add_argument(
            "--live", action="store_true",
            help="For 'list': show in-memory current alerts instead of DB history.",
        )
        # verdict and status use positional args via a small pseudo-protocol:
        # they are added as a single 'rest' nargs='*' list and parsed in
        # handle(), because each sub-action has a different arity.  The
        # argument is named 'rest' (not 'args') to avoid clashing with the
        # BaseCommand.handle(*args, **options) signature — Django would
        # otherwise swallow positional args under the 'args' key.
        parser.add_argument(
            "rest", nargs="*",
            help="verdict: <channel> <alert_ts> <verdict>. "
                 "status: <alert_id> <status>.",
        )
        parser.add_argument(
            "--format", choices=FORMAT_CHOICES, default="text", help=FORMAT_HELP,
        )

    def handle(self, *args, **options) -> None:
        action = options["action"]
        fmt = options["format"]
        if action == "list":
            self._handle_list(options["limit"], options["live"], fmt)
        elif action == "verdict":
            self._handle_verdict(options["rest"], fmt)
        elif action == "status":
            self._handle_status(options["rest"], fmt)

    # ── list ──────────────────────────────────────────────────────────────

    def _handle_list(self, limit: int, live: bool, fmt: str) -> None:
        c = get_container()
        if live:
            alerts = c.alert_service.list(limit)
            source = "live (in-memory)"
        else:
            alerts = c.sqlite.query_alerts(limit)
            source = "history (SQLite)"
        payload = ok_payload(
            source=source,
            threshold=c.alert_service.threshold,
            count=len(alerts),
            alerts=alerts,
        )
        self._emit(payload, fmt)

    # ── verdict ───────────────────────────────────────────────────────────

    def _handle_verdict(self, argv: list[str], fmt: str) -> None:
        if len(argv) != 3:
            self._emit(
                error_payload(
                    "verdict needs: alert verdict <channel> <alert_ts> <verdict>. "
                    f"Valid verdicts: {', '.join(_VALID_VERDICTS)}."
                ),
                fmt,
            )
            return
        channel, ts_str, verdict = argv
        if verdict not in _VALID_VERDICTS:
            self._emit(
                error_payload(
                    f"Invalid verdict '{verdict}'. Valid: {', '.join(_VALID_VERDICTS)}."
                ),
                fmt,
            )
            return
        try:
            alert_ts = float(ts_str)
        except ValueError:
            self._emit(error_payload(f"alert_ts must be a number, got: {ts_str}"), fmt)
            return
        c = get_container()
        ok = c.sqlite.update_alert_verdict(channel, alert_ts, verdict, is_llm=False)
        if not ok:
            self._emit(
                not_found_payload(
                    "channel", channel, alert_ts=alert_ts,
                    message="No matching alert record (wrong channel/ts or SQLite disabled).",
                ),
                fmt,
            )
            return
        self._emit(ok_payload(channel=channel, alert_ts=alert_ts, human_verdict=verdict), fmt)

    # ── status ────────────────────────────────────────────────────────────

    def _handle_status(self, argv: list[str], fmt: str) -> None:
        if len(argv) != 2:
            self._emit(
                error_payload(
                    "status needs: alert status <alert_id> <status>. "
                    f"Valid statuses: {', '.join(_VALID_STATUSES)}."
                ),
                fmt,
            )
            return
        id_str, status = argv
        if status not in _VALID_STATUSES:
            self._emit(
                error_payload(
                    f"Invalid status '{status}'. Valid: {', '.join(_VALID_STATUSES)}."
                ),
                fmt,
            )
            return
        try:
            alert_id = int(id_str)
        except ValueError:
            self._emit(error_payload(f"alert_id must be an integer, got: {id_str}"), fmt)
            return
        c = get_container()
        ok = c.sqlite.update_alert_status(alert_id, status)
        if not ok:
            self._emit(
                not_found_payload("alert_id", alert_id,
                                  message="No matching alert record or SQLite disabled."),
                fmt,
            )
            return
        # Field is 'new_status' (not 'status') to avoid colliding with the
        # top-level {"status": "ok"} marker that ok_payload injects.
        self._emit(ok_payload(id=alert_id, new_status=status), fmt)

    # ── output ────────────────────────────────────────────────────────────

    def _emit(self, payload: dict, fmt: str) -> None:
        if fmt == "json":
            emit(self.stdout, payload, fmt)
        else:
            self._render_text(payload)

    def _render_text(self, payload: dict) -> None:
        status = payload.get("status", "ok")
        if status == "error":
            self.stderr.write(self.style.ERROR(f"Error: {payload.get('message', '')}"))
            return
        if status == "not_found":
            self.stderr.write(self.style.WARNING(f"Not found: {payload.get('message', '')}"))
            return
        if "alerts" in payload:
            self._render_list(payload)
        elif "human_verdict" in payload:
            self.stdout.write(self.style.SUCCESS(
                f"Annotated {payload['channel']} @ {payload['alert_ts']} "
                f"→ human_verdict={payload['human_verdict']}"
            ))
        elif "new_status" in payload and "id" in payload:
            self.stdout.write(self.style.SUCCESS(
                f"Alert #{payload['id']} status → {payload['new_status']}"
            ))

    def _render_list(self, payload: dict) -> None:
        alerts = payload.get("alerts") or []
        self.stdout.write(self.style.SUCCESS(
            f"Alerts ({payload['source']}, threshold={payload['threshold']:.3f})"
        ))
        if not alerts:
            self.stdout.write("  (no alerts)")
            return
        # Column header
        self.stdout.write(
            f"  {'ID':<5} {'Channel':<14} {'Type':<10} {'Score':>6} "
            f"{'Status':<11} {'LLM':<12} {'Human':<12}"
        )
        for a in alerts:
            self.stdout.write(
                f"  {a.get('id', '-'):<5} {str(a.get('channel', ''))[:14]:<14} "
                f"{str(a.get('alert_type', ''))[:10]:<10} "
                f"{a.get('score', 0):>6.2f} {str(a.get('status', ''))[:11]:<11} "
                f"{str(a.get('llm_verdict') or '(未标注)')[:12]:<12} "
                f"{str(a.get('human_verdict') or '(未标注)')[:12]:<12}"
            )
        self.stdout.write(f"  ({payload['count']} shown)")
