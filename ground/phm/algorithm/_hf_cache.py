"""Resolve HuggingFace model ids to local cache snapshot paths.

The TSPulse/TTM-R3 loaders default to the hub id (e.g. ``ibm-research/ttm-r3``)
which makes ``from_pretrained`` / ``get_model`` contact huggingface.co even when
the model is already cached locally.  When that network call fails (SSL cert
errors on this machine) transformers falls back to a meta-tensor placeholder,
which later raises ``NotImplementedError: Cannot copy out of meta tensor`` on
the first ``.to()`` call.

This module resolves a hub id to its on-disk snapshot directory under
``HF_HOME`` (defaulting to ``src/.hf_cache``) so the loaders can pass a real
local path and skip the network entirely.

It also exposes a process-wide lock (``model_load_lock``) that serialises all
``from_pretrained`` calls.  transformers' ``from_pretrained`` uses module-level
initialisation hooks (``init_empty_weights`` / no-init contexts) that mutate
global torch state and are **not thread-safe** — concurrent calls from the
eval thread pool corrupt each other and produce meta tensors.  Wrapping every
loader call in this lock makes model construction serial (one-time, ~1s) while
inference stays fully parallel.

Usage::

    from ._hf_cache import resolve_local_model_path, model_load_lock
    path = resolve_local_model_path("ibm-research/ttm-r3")
    with model_load_lock:
        model = TinyTimeMixerForPrediction.from_pretrained(path, ...)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

__all__ = ["resolve_local_model_path", "ensure_offline_env", "model_load_lock"]

# Process-wide lock: serialises ALL from_pretrained / model-construction calls.
# transformers' module-init hooks are not thread-safe; concurrent construction
# corrupts torch's parameter placeholder state and yields meta tensors.
model_load_lock = threading.Lock()


def _hf_home() -> Path:
    """Return the HF_HOME directory (where the hub cache lives)."""
    env = os.environ.get("HF_HOME", "")
    if env:
        return Path(env)
    # Fall back to the project's conventional cache location:
    # this file is at src/ground/phm/algorithm/_hf_cache.py → src/.hf_cache
    here = Path(__file__).resolve().parent
    return here.parent.parent.parent / ".hf_cache"


def ensure_offline_env() -> None:
    """Set HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE if a local cache exists.

    Called eagerly by the loaders so that even ``from_pretrained(hub_id)``
    stays offline when the model is cached locally.  Idempotent — respects
    values already set by the caller / environment.
    """
    cache = _hf_home() / "hub"
    if cache.is_dir():
        os.environ.setdefault("HF_HOME", str(_hf_home()))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_local_model_path(hub_id: str) -> str | None:
    """Return the local snapshot directory for a HF hub id, or None.

    ``hub_id`` like ``"ibm-research/ttm-r3"`` maps to
    ``<HF_HOME>/hub/models--ibm-research--ttm-r3/snapshots/<sha>/``.  If the
    snapshot has a ``config.json`` (the minimum for from_pretrained) it is
    considered present and the path is returned.  Returns None if not cached.
    """
    if not hub_id or os.path.isdir(hub_id):
        # Already a local path — leave it alone.
        return hub_id if hub_id else None
    # Hub id → cache dir name: "ibm-research/ttm-r3" → "models--ibm-research--ttm-r3"
    cache_name = "models--" + hub_id.replace("/", "--")
    model_cache = _hf_home() / "hub" / cache_name
    snapshots = model_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    # Pick the first snapshot that has a config.json (there is normally one).
    for entry in sorted(snapshots.iterdir()):
        if entry.is_dir() and (entry / "config.json").exists():
            return str(entry)
    return None
