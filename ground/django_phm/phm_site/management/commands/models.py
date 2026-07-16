"""Agent-friendly CLI to enumerate registered models.

Lists every model the system knows about (detector / forecaster / RUL) with
its HuggingFace hub id, context/prediction lengths, and provenance notes.
Pure metadata — does not import torch or load any weights, so it is fast and
safe to run any time.

Usage::

    manage.py models
    manage.py models --format json
    manage.py models --key tspulse
"""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from phm.algorithm import MODEL_REGISTRY, get_model_entry


class Command(BaseCommand):
    help = "List registered PHM models (detector / forecaster / RUL)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--key",
            type=str,
            default="",
            help="Show only the model with this registry key (e.g. tspulse).",
        )
        parser.add_argument(
            "--format",
            choices=["text", "json"],
            default="text",
            help="Output format (default: text; json for agents).",
        )

    def handle(self, *args, **options) -> None:
        key = options["key"]
        fmt = options["format"]
        if key:
            entry = get_model_entry(key)
            if entry is None:
                self.stderr.write(self.style.ERROR(f"Unknown model key: {key}"))
                return
            payload = {"model": _entry_to_dict(entry)}
        else:
            payload = {
                "models": [
                    _entry_to_dict(e) for e in MODEL_REGISTRY.values()
                ]
            }
        if fmt == "json":
            self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            self._render_text(payload)

    def _render_text(self, payload: dict) -> None:
        models = payload.get("models") or [payload["model"]]
        self.stdout.write(self.style.SUCCESS(f"Registered models ({len(models)})"))
        for m in models:
            hub = m["hub_id"] or "(local weights)"
            self.stdout.write(
                f"  {m['key']:<10} [{m['kind']:<10}] {hub}\n"
                f"             context={m['context_length']}, "
                f"prediction={m['prediction_length']}"
            )
            if m.get("notes"):
                self.stdout.write(f"             {m['notes']}")


def _entry_to_dict(entry) -> dict:
    return {
        "key": entry.key,
        "kind": entry.kind,
        "hub_id": entry.hub_id,
        "context_length": entry.context_length,
        "prediction_length": entry.prediction_length,
        "notes": entry.notes,
    }
