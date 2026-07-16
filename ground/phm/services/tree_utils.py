"""Device-tree recursive helpers (pure functions, no side effects).

The device tree stored in ``device_config.json`` is now a *nested* structure:
top-level nodes can be ``folder`` (with ``children``) or ``sensor`` (with a
``sourceId``).  These helpers walk that tree so callers don't need to
re-implement recursion (the previous flat-only code only read top-level
``sourceId`` and silently ignored sensors inside folders).

All functions are tolerant of the legacy *flat* layout (no folders) and of
malformed nodes (missing ``type`` / ``sourceId`` / ``children``).  They never
raise — they just return an empty result for the missing branch.
"""

from __future__ import annotations

from typing import Any

Node = dict[str, Any]


def _is_sensor(node: Node) -> bool:
    """A node is a sensor if it carries a ``sourceId`` (or is explicitly typed)."""
    if not isinstance(node, dict):
        return False
    if node.get("sourceId"):
        return True
    return node.get("type") == "sensor"


def _is_folder(node: Node) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") == "folder":
        return True
    # Infer from children presence — a node with a non-empty children list
    # behaves as a folder even without an explicit type.
    children = node.get("children")
    return isinstance(children, list) and len(children) > 0


def get_flat_sensors(tree: list[Node]) -> list[Node]:
    """Recursively collect every sensor node in the tree (depth-first).

    ``tree`` is the ``device_tree`` array.  Returns a *flat* list of sensor
    node dicts (the same dict objects, not copies).  Legacy flat configs
    (sensors directly at top level) are handled — they are yielded directly.
    """
    out: list[Node] = []

    def walk(nodes: list[Node]) -> None:
        if not isinstance(nodes, list):
            return
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if _is_sensor(n):
                out.append(n)
            if _is_folder(n):
                walk(n.get("children") or [])

    walk(tree)
    return out


def get_folders(tree: list[Node]) -> list[Node]:
    """Return every folder node in the tree (recursive, includes nested)."""
    out: list[Node] = []

    def walk(nodes: list[Node]) -> None:
        if not isinstance(nodes, list):
            return
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if _is_folder(n):
                out.append(n)
                walk(n.get("children") or [])

    walk(tree)
    return out


def get_sensor_to_folder(tree: list[Node]) -> dict[str, str]:
    """Map ``channelName -> folder_id`` for sensors that live inside a folder.

    Only sensors that are direct or indirect descendants of a folder appear in
    the map.  Orphan sensors (top-level, no folder parent) are intentionally
    *omitted* — callers detect them with ``channel_name in result``.
    """
    mapping: dict[str, str] = {}

    def walk_folder(folder: Node) -> None:
        fid = folder.get("id")
        children = folder.get("children") or []
        for c in children:
            if not isinstance(c, dict):
                continue
            if _is_sensor(c) and not _is_folder(c):
                ch = c.get("channelName")
                if ch and fid:
                    mapping[ch] = fid
            if _is_folder(c):
                walk_folder(c)

    for n in tree:
        if isinstance(n, dict) and _is_folder(n):
            walk_folder(n)
    return mapping


def _find_node_by_id(tree: list[Node], node_id: str) -> Node | None:
    """Recursively find a node (folder or sensor) by id; None if not found."""
    for n in tree:
        if not isinstance(n, dict):
            continue
        if n.get("id") == node_id:
            return n
        if _is_folder(n):
            found = _find_node_by_id(n.get("children") or [], node_id)
            if found is not None:
                return found
    return None


def remove_node(tree: list[Node], node_id: str) -> bool:
    """Remove the node with ``node_id`` from ``tree`` in place.

    Walks the tree depth-first; the first match is deleted and the function
    returns ``True``.  Removing a folder also discards its ``children``.
    Returns ``False`` if no node with that id exists.

    The list is mutated in place — callers that need the original tree
    should pass a deep copy.  This matches the semantics of
    :func:`list.remove` and keeps the helper allocation-free for the
    common ``load → modify → save`` CLI flow.
    """
    for i, n in enumerate(tree):
        if not isinstance(n, dict):
            continue
        if n.get("id") == node_id:
            del tree[i]
            return True
    for n in tree:
        if isinstance(n, dict) and _is_folder(n):
            if remove_node(n.get("children") or [], node_id):
                return True
    return False


def get_sensors_in_folder(tree: list[Node], folder_id: str) -> list[Node]:
    """Return all sensor nodes under the folder with ``folder_id`` (recursive).

    If ``folder_id`` refers to a sensor (not a folder) or doesn't exist,
    returns ``[]`` — this enforces "folder-only" semantics so a caller that
    accidentally passes a sensor id gets an empty result rather than that
    sensor echoed back.
    """
    target = _find_node_by_id(tree, folder_id)
    if target is None or not _is_folder(target):
        return []
    return get_flat_sensors([target])


def get_aggregation_strategy(config: dict[str, Any]) -> str:
    """Return the configured aggregation strategy (default ``"min"``).

    Reads the top-level ``aggregation_strategy`` key from the config dict.
    Accepts ``"min"`` or ``"mean"``; anything else falls back to ``"min"``
    so a malformed config never breaks health aggregation.
    """
    strat = config.get("aggregation_strategy", "min") if isinstance(config, dict) else "min"
    return strat if strat in ("min", "mean") else "min"


__all__ = [
    "get_flat_sensors",
    "get_folders",
    "get_sensor_to_folder",
    "get_sensors_in_folder",
    "get_aggregation_strategy",
    "remove_node",
]
