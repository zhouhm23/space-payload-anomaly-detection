"""Device-tree config persistence service.

Mirrors the legacy ``server.py`` behaviour: read/write the JSON file at
``device_config.json`` and push updates to the space segment over TCP.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_GROUND_DIR = _HERE.parent.parent  # src/ground
if str(_GROUND_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUND_DIR))

from comm import GroundClient  # noqa: E402


class ConfigService:
    def __init__(
        self,
        config_path: Path,
        space_host: str = "127.0.0.1",
        space_port: int = 9876,
    ) -> None:
        self.config_path = config_path
        self.space_host = space_host
        self.space_port = space_port

    def load(self) -> dict:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to read config %s: %s", self.config_path, e)
        # Default skeleton — always carries a device_tree and aggregation_strategy
        # so downstream consumers (HealthService, RulService) never miss the key.
        return {"device_tree": [], "aggregation_strategy": "min"}

    def save(self, body: dict) -> dict:
        # Preserve aggregation_strategy if the frontend omits it (older
        # clients POST only {device_tree: [...]} and would otherwise wipe
        # the key).  Default to "min" per Slice 0 spec.
        if "aggregation_strategy" not in body:
            body["aggregation_strategy"] = "min"
        # Defensive uniqueness check: duplicate sourceId would make two
        # sensors share the same ring-buffer channel and SQLite table,
        # silently corrupting data.  The frontend also checks this, but a
        # client can bypass it.
        dup = self._find_duplicate_source(body.get("device_tree", []))
        if dup:
            return {"status": "error", "message": f"重复的数据源标识: {dup}"}
        self.config_path.write_text(
            json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Push tree to space segment via TCP (best-effort)
        try:
            client = GroundClient(host=self.space_host, port=self.space_port, timeout=2)
            client.poll({"device_tree": body.get("device_tree", [])})
        except Exception:
            pass
        return {"status": "ok"}

    @staticmethod
    def _find_duplicate_source(tree: list) -> str | None:
        """Return the first duplicate sourceId in the tree, or None."""
        seen: set[str] = set()

        def walk(nodes):
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                if n.get("type") == "sensor":
                    sid = n.get("sourceId") or n.get("source_id")
                    if sid:
                        if sid in seen:
                            return sid
                        seen.add(sid)
                children = n.get("children")
                if children:
                    found = walk(children)
                    if found:
                        return found
            return None

        return walk(tree)


__all__ = ["ConfigService"]
