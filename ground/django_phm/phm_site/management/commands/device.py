"""Agent-friendly CLI for the device tree (device_config.json).

Mirrors what the front-end config panel does over HTTP — but scriptable.
All three sub-actions go through ``ConfigService`` (same instance the HTTP
API uses), so validation (empty-tree refusal, duplicate-sourceId check)
and the TCP push to the space segment happen exactly once, in one place.

Usage::

    # Inspect the current tree (read-only, safe anytime)
    manage.py device show
    manage.py device show --format json

    # Replace the whole tree from a JSON file (destructive — needs --confirm)
    manage.py device save path/to/tree.json --confirm
    manage.py device save tree.json --confirm --format json

    # Remove one node by id (destructive — needs --confirm)
    manage.py device rm <node-id> --confirm
    manage.py device rm F1-3 --confirm --format json

The ``--confirm`` flag is mandatory for ``save`` and ``rm`` because both
overwrite ``device_config.json`` on disk.  ``show`` is read-only and needs
no confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from phm.services.tree_utils import get_flat_sensors, get_folders, remove_node
from phm_site.services_bridge import get_container

from ._common import (
    FORMAT_CHOICES,
    FORMAT_HELP,
    emit,
    error_payload,
    not_found_payload,
    ok_payload,
)


class Command(BaseCommand):
    help = "Inspect or modify the device tree (show / save / rm)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "action",
            choices=["show", "save", "rm"],
            help="show: print the current tree. "
                 "save: replace the whole tree from a JSON file. "
                 "rm: delete a single node by id.",
        )
        parser.add_argument(
            "target",
            nargs="?",
            default="",
            help="For 'save': path to the JSON file. "
                 "For 'rm': the node id to remove. "
                 "Ignored for 'show'.",
        )
        parser.add_argument(
            "--confirm",
            action="store_true",
            help="Required for 'save' and 'rm' (both overwrite device_config.json).",
        )
        parser.add_argument(
            "--format",
            choices=FORMAT_CHOICES,
            default="text",
            help=FORMAT_HELP,
        )

    def handle(self, *args, **options) -> None:
        action = options["action"]
        fmt = options["format"]
        if action == "show":
            self._handle_show(fmt)
        elif action == "save":
            self._handle_save(options["target"], options["confirm"], fmt)
        elif action == "rm":
            self._handle_rm(options["target"], options["confirm"], fmt)

    # ── show ──────────────────────────────────────────────────────────────

    def _handle_show(self, fmt: str) -> None:
        cfg = get_container().config
        data = cfg.load()
        tree = data.get("device_tree", [])
        payload = ok_payload(
            source=str(cfg.config_path),
            aggregation_strategy=data.get("aggregation_strategy", "min"),
            folders=len(get_folders(tree)),
            sensors=len(get_flat_sensors(tree)),
            tree=tree,
        )
        self._emit(payload, fmt)

    # ── save ──────────────────────────────────────────────────────────────

    def _handle_save(self, target: str, confirm: bool, fmt: str) -> None:
        if not confirm:
            self._emit(
                error_payload(
                    "save overwrites device_config.json — re-run with --confirm to proceed."
                ),
                fmt,
            )
            return
        if not target:
            self._emit(
                error_payload("save needs a JSON file path: device save <file.json> --confirm."),
                fmt,
            )
            return
        path = Path(target)
        if not path.is_file():
            self._emit(error_payload(f"File not found: {target}"), fmt)
            return
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self._emit(error_payload(f"Failed to parse JSON: {e}"), fmt)
            return
        cfg = get_container().config
        result = cfg.save(body)
        if result.get("status") != "ok":
            # ConfigService already shaped the error body (message + maybe current_tree)
            self._emit(result, fmt)
            return
        tree = body.get("device_tree", [])
        self._emit(
            ok_payload(
                source=target,
                folders=len(get_folders(tree)),
                sensors=len(get_flat_sensors(tree)),
            ),
            fmt,
        )

    # ── rm ────────────────────────────────────────────────────────────────

    def _handle_rm(self, target: str, confirm: bool, fmt: str) -> None:
        if not confirm:
            self._emit(
                error_payload(
                    "rm rewrites device_config.json — re-run with --confirm to proceed."
                ),
                fmt,
            )
            return
        if not target:
            self._emit(error_payload("rm needs a node id: device rm <node-id> --confirm."), fmt)
            return
        cfg = get_container().config
        data = cfg.load()
        tree = data.get("device_tree", [])
        if not remove_node(tree, target):
            self._emit(not_found_payload("node_id", target), fmt)
            return
        result = cfg.save({"device_tree": tree,
                           "aggregation_strategy": data.get("aggregation_strategy", "min")})
        if result.get("status") != "ok":
            self._emit(result, fmt)
            return
        self._emit(
            ok_payload(removed=target,
                       remaining_folders=len(get_folders(tree)),
                       remaining_sensors=len(get_flat_sensors(tree))),
            fmt,
        )

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
            self.stderr.write(self.style.WARNING(
                f"Node not found: {payload.get('node_id')}"
            ))
            return
        action = payload.get("source")  # 'source' only present on show/save
        if "removed" in payload:
            self._render_rm(payload)
        elif action and "tree" in payload:
            self._render_show(payload)
        elif action:
            self._render_save(payload)

    def _render_show(self, payload: dict) -> None:
        self.stdout.write(self.style.SUCCESS(
            f"Device tree (source: {payload['source']})"
        ))
        self.stdout.write(f"  aggregation_strategy: {payload['aggregation_strategy']}")
        self.stdout.write(
            f"  Folders: {payload['folders']}, Sensors: {payload['sensors']}"
        )
        tree = payload.get("tree") or []
        if tree:
            self.stdout.write("")
            self._render_tree_nodes(tree, indent=2)

    def _render_tree_nodes(self, nodes, indent: int) -> None:
        pad = " " * indent
        for n in nodes:
            if not isinstance(n, dict):
                continue
            ntype = "folder" if n.get("type") == "folder" or n.get("children") else "sensor"
            nid = n.get("id", "?")
            name = n.get("name", "")
            if ntype == "folder":
                self.stdout.write(f"{pad}[F] {nid}  {name}".rstrip())
                self._render_tree_nodes(n.get("children") or [], indent + 2)
            else:
                src = n.get("sourceId", "")
                ch = n.get("channelName", "")
                blk = n.get("blockSize", "")
                self.stdout.write(f"{pad}[S] {nid}  {name}  ({src}, ch={ch}, block={blk})".rstrip())

    def _render_save(self, payload: dict) -> None:
        self.stdout.write(self.style.SUCCESS(
            f"Saved device tree from {payload['source']}"
        ))
        self.stdout.write(
            f"  Folders: {payload['folders']}, Sensors: {payload['sensors']}"
        )

    def _render_rm(self, payload: dict) -> None:
        self.stdout.write(self.style.SUCCESS(
            f"Removed node {payload['removed']}"
        ))
        self.stdout.write(
            f"  Remaining: {payload['remaining_folders']} folders, "
            f"{payload['remaining_sensors']} sensors"
        )
