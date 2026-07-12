"""Health-value service.

Implements the fixed health formula required by the task spec:

    单通道健康值 = 异常分数 ≤ THRESHOLD 的正常点数 / 总采样点数 × 100
    系统总健康值 = 全部采集通道健康值平均值

When a ``ConfigService`` is injected, the service also returns a
``folders`` map aggregating per-channel health up to each device-tree
folder (``min`` strategy = "木桶效应" / worst sensor wins; ``mean`` = average).
"""

from __future__ import annotations

from typing import Iterable

from ..config import ANOMALY_THRESHOLD
from ..database import RingBuffer
from .tree_utils import (
    get_aggregation_strategy,
    get_folders,
    get_sensors_in_folder,
)

try:  # ConfigService lives in the same package; import lazily to avoid cycles
    from .config_service import ConfigService
except Exception:  # pragma: no cover — defensive
    ConfigService = None  # type: ignore[assignment, misc]


def channel_health(scores: Iterable[float], threshold: float = ANOMALY_THRESHOLD) -> float:
    """Single-channel health in [0, 100]."""
    scores = list(scores)
    if not scores:
        return 100.0
    normal = sum(1 for s in scores if s <= threshold)
    return round(normal / len(scores) * 100.0, 1)


class HealthService:
    def __init__(self, ring: RingBuffer, config_service: "ConfigService | None" = None) -> None:
        self.ring = ring
        self.config_service = config_service

    def system_health(self, block_size: int = 20000) -> dict:
        """Return per-channel + aggregate + folder-aggregated health snapshot."""
        per_channel_scores = self.ring.all_channel_scores(block_size)
        per_channel: dict[str, float] = {}
        for ch, scores in per_channel_scores.items():
            per_channel[ch] = channel_health(scores)
        if per_channel:
            system = round(sum(per_channel.values()) / len(per_channel), 1)
        else:
            system = 100.0

        result: dict = {
            "system": system,
            "channels": per_channel,
            "threshold": ANOMALY_THRESHOLD,
        }

        # Folder aggregation — only when config is available.  Sensors whose
        # channel is not yet in the ring (no data) are skipped silently.
        folders = self._aggregate_folders(per_channel)
        if folders is not None:
            result["folders"] = folders

        return result

    def _aggregate_folders(self, per_channel: dict[str, float]) -> dict[str, dict] | None:
        """Build {folder_id: {name, health, strategy, channels}} from the tree.

        Returns ``None`` when no ConfigService is wired (legacy callers) so the
        ``folders`` key is simply omitted from the response rather than being
        an empty dict — keeps the contract additive and backward-compatible.
        """
        if self.config_service is None:
            return None
        config = self.config_service.load()
        tree = config.get("device_tree", [])
        strategy = get_aggregation_strategy(config)

        out: dict[str, dict] = {}
        for folder in get_folders(tree):
            sensors = get_sensors_in_folder(tree, folder.get("id", ""))
            ch_names = [s.get("channelName") for s in sensors if s.get("channelName")]
            ch_healths = [per_channel[ch] for ch in ch_names if ch in per_channel]
            if not ch_healths:
                continue  # folder has no sensors with data yet — skip it
            if strategy == "mean":
                value = round(sum(ch_healths) / len(ch_healths), 1)
            else:  # "min" default — 木桶效应, worst sensor wins
                value = round(min(ch_healths), 1)
            out[folder["id"]] = {
                "name": folder.get("name", folder.get("id", "")),
                "health": value,
                "strategy": strategy,
                "channels": [ch for ch in ch_names if ch in per_channel],
            }
        return out


__all__ = ["HealthService", "channel_health"]
