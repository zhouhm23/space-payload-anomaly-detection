"""Agent-friendly CLI for the RUL (Remaining Useful Life) service.

Usage::

    # Show what models are loaded and which channels are tagged.
    manage.py rul status
    manage.py rul status --format json

    # Predict RUL for one channel (ad-hoc, does not advance the pointer).
    manage.py rul predict --channel CMAPSS_FD001_1
    manage.py rul predict --channel CMAPSS_FD001_1 --format json

    # Advance one cycle and predict every tagged channel (same as /api/rul).
    manage.py rul predict --all

The command imports the shared service container (same one the HTTP API
uses), so the output is always consistent with what the front-end sees.
"""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from phm_site.services_bridge import get_container


class Command(BaseCommand):
    help = "Query the RUL degradation-prediction service (status / sources / predict)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "action",
            choices=["status", "predict"],
            default="status",
            nargs="?",
            help="status (default): show loaded models + tagged channels. "
                 "predict: run a prediction.",
        )
        parser.add_argument(
            "--channel",
            type=str,
            default="",
            help="Channel name for a single prediction (e.g. CMAPSS_FD001_1).",
        )
        parser.add_argument(
            "--all",
            dest="predict_all",
            action="store_true",
            help="With 'predict': advance one cycle and predict every tagged "
                 "channel (mirrors GET /api/rul).",
        )
        parser.add_argument(
            "--format",
            choices=["text", "json"],
            default="text",
            help="Output format (default: text; json for agents).",
        )

    def handle(self, *args, **options) -> None:
        c = get_container()
        if c.rul is None:
            self._emit(
                {"status": "disabled",
                 "message": "RUL service is not running (assets missing or "
                            "RUL_ENABLED=False). See startup logs."},
                options["format"],
            )
            return

        action = options["action"]
        if action == "status":
            self._emit(c.rul.status(), options["format"])
        elif action == "predict":
            self._handle_predict(c, options)
        else:  # pragma: no cover — argparse choices guard this
            raise CommandError(f"Unknown action: {action}")

    # ── helpers ────────────────────────────────────────────────────────

    def _handle_predict(self, container, options) -> None:
        fmt = options["format"]
        channel = options["channel"]
        predict_all = options["predict_all"]
        if predict_all:
            results = container.rul.predict_all()
            self._emit({"status": "ok", "count": len(results), "data": results}, fmt)
            return
        if not channel:
            raise CommandError(
                "predict needs --channel <name> or --all. "
                "Use 'manage.py rul status' to list tagged channels."
            )
        result = container.rul.predict(channel)
        if result is None:
            self._emit(
                {"status": "not_found",
                 "channel": channel,
                 "message": "Channel is not tagged for RUL or has no data window."},
                fmt,
            )
            return
        self._emit({"status": "ok", "data": result}, fmt)

    def _emit(self, payload: dict, fmt: str) -> None:
        if fmt == "json":
            self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            self._render_text(payload)

    def _render_text(self, payload: dict) -> None:
        status = payload.get("status", "ok")
        if status == "disabled":
            self.stdout.write(self.style.WARNING(
                f"RUL service disabled: {payload.get('message', '')}"
            ))
            return
        if status == "not_found":
            self.stdout.write(self.style.WARNING(
                f"Channel {payload.get('channel')} not available for RUL."
            ))
            return
        if "enabled_models" in payload:
            self._render_status(payload)
        elif "data" in payload and isinstance(payload["data"], list):
            self._render_results(payload["data"])
        elif "data" in payload:
            self._render_results([payload["data"]])

    def _render_status(self, payload: dict) -> None:
        self.stdout.write(self.style.SUCCESS("RUL service status"))
        models = payload.get("enabled_models", [])
        self.stdout.write(f"  Enabled models: {', '.join(models) or '(none)'}")
        sources = payload.get("data_sources", {})
        for mid, channels in sources.items():
            self.stdout.write(
                f"  Data source [{mid}]: {len(channels)} channels "
                f"({channels[0] if channels else '—'}..)"
            )
        tagged = payload.get("tagged_channels", {})
        self.stdout.write(f"  Tagged channels: {len(tagged)}")
        for ch, mid in tagged.items():
            self.stdout.write(f"    {ch} → {mid}")
        self.stdout.write(
            f"  Window: {payload.get('window_cycles')} cycles, "
            f"history: {payload.get('history_len')}"
        )

    def _render_results(self, results: list[dict]) -> None:
        if not results:
            self.stdout.write("(no results)")
            return
        for r in results:
            self.stdout.write(
                f"  {r['channel']:<24} RUL={r['rul']:>6} {r['unit']}  "
                f"({r['model']}, max={r['max_rul']})"
            )
