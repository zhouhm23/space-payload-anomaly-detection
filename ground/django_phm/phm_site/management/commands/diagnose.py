"""Agent-friendly CLI for LLM diagnosis (run / auto / status / list).

Mirrors four HTTP endpoints:
  * ``POST /api/diagnosis``            — manual single diagnosis
  * ``POST /api/diagnosis/auto``       — batch auto-diagnosis (background)
  * ``GET  /api/diagnosis/auto/status``— auto-diagnosis progress
  * ``GET  /api/diagnosis/done``       — completed diagnoses list

Diagnosis triggers need no ``--confirm`` flag: the LLM call is idempotent
(cached per ``(channel, alert_type, alert_ts)``) and ``auto`` runs in a
background daemon thread, so re-running is safe.

Usage::

    # Manual single diagnosis (uses cache unless --force)
    manage.py diagnose run <channel>
    manage.py diagnose run C-1 --type predicted --ts 1752633600
    manage.py diagnose run C-1 --force --format json

    # Batch auto-diagnosis (starts a background thread)
    manage.py diagnose auto
    manage.py diagnose auto --format json

    # Query progress of the running auto-diagnosis
    manage.py diagnose status

    # List already-completed diagnoses (cached in SQLite)
    manage.py diagnose list
    manage.py diagnose list --limit 50 --format json
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from phm_site.services_bridge import get_container

from ._common import (
    FORMAT_CHOICES,
    FORMAT_HELP,
    emit,
    error_payload,
    ok_payload,
)


class Command(BaseCommand):
    help = "Trigger or query LLM diagnosis (run / auto / status / list)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "action",
            choices=["run", "auto", "status", "list"],
            help="run: single diagnosis. auto: batch background. "
                 "status: auto progress. list: completed diagnoses.",
        )
        parser.add_argument(
            "channel", nargs="?", default="",
            help="For 'run': the channel name (e.g. C-1).",
        )
        parser.add_argument(
            "--type", dest="alert_type", default="measured",
            choices=["measured", "predicted"],
            help="For 'run': alert type (default measured).",
        )
        parser.add_argument(
            "--ts", dest="alert_ts", default=None,
            help="For 'run': alert timestamp (cache key).",
        )
        parser.add_argument(
            "--force", action="store_true",
            help="For 'run': bypass cache and re-run the LLM.",
        )
        parser.add_argument(
            "--limit", type=int, default=50,
            help="For 'list': max entries (default 50).",
        )
        parser.add_argument(
            "--format", choices=FORMAT_CHOICES, default="text", help=FORMAT_HELP,
        )

    def handle(self, *args, **options) -> None:
        action = options["action"]
        fmt = options["format"]
        if action == "run":
            self._handle_run(options["channel"], options["alert_type"],
                             options["alert_ts"], options["force"], fmt)
        elif action == "auto":
            self._handle_auto(fmt)
        elif action == "status":
            self._handle_status(fmt)
        elif action == "list":
            self._handle_list(options["limit"], fmt)

    # ── run ───────────────────────────────────────────────────────────────

    def _handle_run(self, channel: str, alert_type: str,
                    alert_ts_str: str | None, force: bool, fmt: str) -> None:
        if not channel:
            self._emit(error_payload("run needs a channel: diagnose run <channel>."), fmt)
            return
        alert_ts: float | None = None
        if alert_ts_str:
            try:
                alert_ts = float(alert_ts_str)
            except ValueError:
                self._emit(error_payload(f"--ts must be a number, got: {alert_ts_str}"), fmt)
                return
        c = get_container()
        result = c.diagnosis.diagnose(channel, alert_type=alert_type,
                                      alert_ts=alert_ts, force_refresh=force)
        # diagnose() always returns a dict (never raises); the 'error' field
        # signals failure.  Wrap into our standard envelope.
        if result.get("error"):
            payload = error_payload(
                result["error"],
                channel=channel,
                alert_type=alert_type,
                cached=result.get("cached", False),
            )
        else:
            payload = ok_payload(**result)
        self._emit(payload, fmt)

    # ── auto ──────────────────────────────────────────────────────────────

    def _handle_auto(self, fmt: str) -> None:
        c = get_container()
        result = c.diagnosis.auto_diagnose_all()
        # auto_diagnose_all returns {"started": bool, "total": int} or
        # {"started": False, "error": "..."}.  Normalise into our envelope.
        if result.get("started"):
            payload = ok_payload(started=True, total=result.get("total", 0))
        else:
            payload = error_payload(result.get("error", "unknown reason"))
        self._emit(payload, fmt)

    # ── status ────────────────────────────────────────────────────────────

    def _handle_status(self, fmt: str) -> None:
        c = get_container()
        progress = c.diagnosis.auto_status
        self._emit(ok_payload(**progress), fmt)

    # ── list ──────────────────────────────────────────────────────────────

    def _handle_list(self, limit: int, fmt: str) -> None:
        c = get_container()
        rows = c.sqlite.list_diagnosis_keys(limit)
        self._emit(ok_payload(count=len(rows), diagnoses=rows), fmt)

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
        # Branch on which action produced this payload.
        if "diagnosis" in payload:
            self._render_run(payload)
        elif "started" in payload:
            self._render_auto(payload)
        elif "running" in payload:
            self._render_status(payload)
        elif "diagnoses" in payload:
            self._render_list(payload)

    def _render_run(self, payload: dict) -> None:
        ch = payload.get("channel", "?")
        atype = payload.get("alert_type", "?")
        verdict = payload.get("llm_verdict") or "(none)"
        cached = payload.get("cached", False)
        elapsed = payload.get("elapsed_sec", 0.0)
        self.stdout.write(self.style.SUCCESS(
            f"Diagnosis for {ch} ({atype})"
        ))
        self.stdout.write(
            f"  Verdict: {verdict}  (cached={cached}, elapsed={elapsed:.1f}s)"
        )
        body = payload.get("diagnosis") or "(empty)"
        self.stdout.write("  ---")
        # Indent each line of the markdown body by 2 spaces for readability.
        for line in str(body).splitlines() or ["(empty)"]:
            self.stdout.write(f"  {line}")

    def _render_auto(self, payload: dict) -> None:
        total = payload.get("total", 0)
        self.stdout.write(self.style.SUCCESS(
            f"Auto-diagnosis started ({total} target(s) queued)"
        ))
        self.stdout.write("  Use 'manage.py diagnose status' to track progress.")

    def _render_status(self, payload: dict) -> None:
        running = payload.get("running", False)
        done = payload.get("done", 0)
        total = payload.get("total", 0)
        errors = payload.get("errors", 0)
        state = "running" if running else "idle"
        self.stdout.write(self.style.SUCCESS(f"Auto-diagnosis: {state}"))
        self.stdout.write(f"  Progress: {done}/{total} done, {errors} errors")

    def _render_list(self, payload: dict) -> None:
        rows = payload.get("diagnoses") or []
        self.stdout.write(self.style.SUCCESS(
            f"Completed diagnoses ({payload.get('count', 0)})"
        ))
        if not rows:
            self.stdout.write("  (none)")
            return
        self.stdout.write(
            f"  {'Channel':<16} {'Type':<10} {'TS':<16} {'LLM verdict':<14}"
        )
        for r in rows:
            self.stdout.write(
                f"  {str(r.get('channel', ''))[:16]:<16} "
                f"{str(r.get('alert_type', ''))[:10]:<10} "
                f"{str(r.get('alert_ts', ''))[:16]:<16} "
                f"{str(r.get('llm_verdict') or '(none)')[:14]:<14}"
            )
