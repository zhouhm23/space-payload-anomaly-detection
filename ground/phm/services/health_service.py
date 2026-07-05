"""Health-value service.

Implements the fixed health formula required by the task spec:

    单通道健康值 = 异常分数 ≤ THRESHOLD 的正常点数 / 总采样点数 × 100
    系统总健康值 = 全部采集通道健康值平均值
"""

from __future__ import annotations

from typing import Iterable

from ..config import ANOMALY_THRESHOLD
from ..database import RingBuffer


def channel_health(scores: Iterable[float], threshold: float = ANOMALY_THRESHOLD) -> float:
    """Single-channel health in [0, 100]."""
    scores = list(scores)
    if not scores:
        return 100.0
    normal = sum(1 for s in scores if s <= threshold)
    return round(normal / len(scores) * 100.0, 1)


class HealthService:
    def __init__(self, ring: RingBuffer) -> None:
        self.ring = ring

    def system_health(self, block_size: int = 20000) -> dict:
        """Return per-channel + aggregate health snapshot."""
        per_channel_scores = self.ring.all_channel_scores(block_size)
        per_channel: dict[str, float] = {}
        for ch, scores in per_channel_scores.items():
            per_channel[ch] = channel_health(scores)
        if per_channel:
            system = round(sum(per_channel.values()) / len(per_channel), 1)
        else:
            system = 100.0
        return {
            "system": system,
            "channels": per_channel,
            "threshold": ANOMALY_THRESHOLD,
        }


__all__ = ["HealthService", "channel_health"]
