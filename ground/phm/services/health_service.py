"""Health-value service.

Implements the single-block health formula required by the v1.0 spec
(``docs/产品需求书.md`` §健康度计算方式):

    单通道健康值 = 1 - 异常点数 / 总点数   （单数据块内，范围 [0, 1]）

  * 异常点 = anomaly_score > THRESHOLD 的采样点
  * 块大小由传感器参数决定（block_size），不做 20000 的隐式默认
  * 范围严格 [0, 1]，1 = 完全健康，0 = 全部异常
  * 文件夹节点 = 子通道 min（保守，木桶效应）或 mean，范围仍 [0, 1]
  * 系统总健康值 = 全部启用通道健康值平均

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
    get_flat_sensors,
    get_folders,
    get_sensors_in_folder,
    is_special_sensor,
)

try:  # ConfigService lives in the same package; import lazily to avoid cycles
    from .config_service import ConfigService
except Exception:  # pragma: no cover — defensive
    ConfigService = None  # type: ignore[assignment, misc]


def channel_health(scores: Iterable[float], threshold: float = ANOMALY_THRESHOLD) -> float:
    """Single-channel health in [0, 1].

    Formula: ``1 - 异常点数 / 总点数``.  Anomaly points are those with
    ``score > threshold`` (strictly greater, mirroring the warning trigger).
    Empty input returns 1.0 (no evidence of anomaly = fully healthy).
    """
    scores = list(scores)
    if not scores:
        return 1.0
    abnormal = sum(1 for s in scores if s is not None and s > threshold)
    return round(1.0 - abnormal / len(scores), 3)


class HealthService:
    def __init__(self, ring: RingBuffer, config_service: "ConfigService | None" = None) -> None:
        self.ring = ring
        self.config_service = config_service

    def system_health(self, block_size: int = 20000) -> dict:
        """Return per-channel + aggregate + folder-aggregated health snapshot.

        ``block_size`` is the number of recent samples per channel to score
        over.  The legacy default (20000) matches ``RING_BUFFER_MAX`` and
        simply means "use whatever is in the ring".  v1.0 callers that want
        single-block semantics (per ``docs/产品需求书.md`` §健康度计算方式)
        should pass the sensor's transport ``blockSize`` (e.g. 512); the
        formula stays the same (``1 - 异常点数 / 总点数``).

        ``@rul``-marked special sensors are excluded from both the system
        aggregate and folder aggregation — they run a separate RUL pipeline
        and would otherwise skew anomaly-based health (Day22 issue 3.1:
        a C-MAPSS special channel dragged its folder to 73% while ordinary
        siblings were 98%/100%).
        """
        per_channel_scores = self.ring.all_channel_scores(block_size)
        excluded = self._special_channel_names()
        per_channel: dict[str, float] = {}
        for ch, scores in per_channel_scores.items():
            if ch in excluded:
                continue  # @rul special channels don't participate in health
            per_channel[ch] = channel_health(scores)
        if per_channel:
            system = round(sum(per_channel.values()) / len(per_channel), 3)
        else:
            system = 1.0

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

    def _special_channel_names(self) -> set[str]:
        """Return the set of ``channelName``s belonging to ``@rul`` special
        sensors, so they can be excluded from health aggregation.

        Returns an empty set when no ConfigService is wired (legacy callers)
        or when the device tree contains no special sensors.
        """
        if self.config_service is None:
            return set()
        config = self.config_service.load()
        tree = config.get("device_tree", [])
        return {
            s.get("channelName")
            for s in get_flat_sensors(tree)
            if is_special_sensor(s) and s.get("channelName")
        }

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
            # Exclude @rul special sensors — they don't participate in health
            # aggregation (separate RUL pipeline, would skew folder health).
            ordinary = [s for s in sensors if not is_special_sensor(s)]
            ch_names = [s.get("channelName") for s in ordinary if s.get("channelName")]
            ch_healths = [per_channel[ch] for ch in ch_names if ch in per_channel]
            if not ch_healths:
                continue  # folder has no sensors with data yet — skip it
            if strategy == "mean":
                value = round(sum(ch_healths) / len(ch_healths), 3)
            else:  # "min" default — 木桶效应, worst sensor wins
                value = round(min(ch_healths), 3)
            out[folder["id"]] = {
                "name": folder.get("name", folder.get("id", "")),
                "health": value,
                "strategy": strategy,
                "channels": [ch for ch in ch_names if ch in per_channel],
            }
        return out


__all__ = ["HealthService", "channel_health"]
