"""Agent-friendly CLI to inspect system runtime configuration.

Surfaces everything backed by ``data/system_config.json`` (network endpoints,
storage sizing, thresholds, feature flags) without requiring a file open.
Read-only — operators and agents can audit the effective config any time.

Usage::

    manage.py config                 # full dump (text)
    manage.py config --format json   # machine-readable
    manage.py config --section rul   # one section only
"""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from phm.services.system_config_service import get_system_config


class Command(BaseCommand):
    help = "Dump the effective system configuration (from system_config.json)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--section",
            type=str,
            default="",
            help="Show only this section (e.g. thresholds, rul, storage).",
        )
        parser.add_argument(
            "--format",
            choices=["text", "json"],
            default="text",
            help="Output format (default: text; json for agents).",
        )

    def handle(self, *args, **options) -> None:
        cfg = get_system_config()
        snapshot = cfg.snapshot()
        section = options["section"]
        fmt = options["format"]

        if section:
            if section not in snapshot:
                self.stderr.write(self.style.ERROR(
                    f"Unknown section: {section}. "
                    f"Available: {', '.join(sorted(snapshot.keys()))}"
                ))
                return
            payload = {section: snapshot[section]}
        else:
            payload = snapshot

        if fmt == "json":
            self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            self._render_text(payload, cfg.config_path)

    def _render_text(self, payload: dict, source_path: str) -> None:
        self.stdout.write(self.style.SUCCESS(
            f"System config (source: {source_path})"
        ))
        for section, values in payload.items():
            self.stdout.write(f"\n[{section}]")
            for key, val in values.items():
                self.stdout.write(f"  {key:<28} = {val}")
