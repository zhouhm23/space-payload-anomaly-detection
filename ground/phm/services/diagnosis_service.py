"""LLM-powered anomaly diagnosis service.

Aggregates the per-channel detection context (three-layer cascade output,
recent telemetry statistics, historical alerts, device-tree position,
offline calibration) into a structured prompt **plus a PNG waveform chart**,
calls an OpenAI-compatible vision-capable chat API, and returns a Markdown
diagnosis report covering root-cause analysis and trend assessment.

The service is **on-demand** — it only runs when a user requests a
diagnosis for a specific alert/warning.  Diagnosis results are cached
in SQLite (``diagnosis_records`` table) keyed by ``(channel, alert_type,
alert_ts)`` so repeated clicks return the stored report without
re-calling the LLM.  A structured ``llm_verdict`` (real / false_alarm /
uncertain) is parsed from the report and written back to the
WarningEntry (predicted) or alert_records row (measured).

Vision-model approach (Day17-续4): the alert-time waveform is rendered as
a 2-panel PNG chart (raw values + anomaly scores with threshold line) and
sent as a base64 image to a vision LLM (GLM-4V-Flash).  Comparative
experiments showed GLM-4V-Flash + PNG achieves 75% accuracy on a mixed
alert set (5 real / 7 false-alarm), versus 0-58% for text-only models.
The vision model can "see" waveform shape (periodicity, amplitude
changes, step jumps) that text statistics cannot convey.

Configuration is via standard environment variables so any
OpenAI-compatible provider works::

    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
    LLM_MODEL=GLM-4V-Flash                           # vision-capable model
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
import time
from typing import Any

import httpx
import numpy as np

# Matplotlib for rendering waveform PNG charts for vision LLM input.
# Use the non-interactive Agg backend — this runs in a server thread.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import (
    ANOMALY_THRESHOLD,
    FORECAST_PREDICTION_LENGTH,
)
from ..algorithm.cascade_types import CascadeOutput

logger = logging.getLogger(__name__)

__all__ = ["DiagnosisService"]


_SYSTEM_PROMPT = """你是航天器有效载荷健康管理（PHM）分析师。你的任务是判断一条告警是真实异常还是误报。

重要：告警已触发不代表一定是真实异常。检测模型（TSPulse 重建误差）会产生误报，尤其在周期性波形、数据漂移、短窗口等情况下。你需要基于数据独立判断，不要预设告警为真。

数据中有用 █▇▆▅▄▃▂▁ 字符表示的波形迷你图（█=高值，▁=低值，从左到右是时间顺序）。请像看图表一样从波形形态判断：原始值波形是否有突变/漂移/异常振幅？异常分数波形是否有明显的尖峰？

判定依据优先级（从高到低）：
1. 异常分数相对正常基线的偏离：若告警时刻分数超出正常段 p95 的 3σ 以上，这是强证据，即使原始值偏离不大也应认真考虑（检测模型能捕获人眼难以察觉的模式变化）。
2. 原始值波形的形态变化：突变、振幅异常、基线漂移比单纯偏离均值更有意义。请直接看波形迷你图判断。
3. 周期性波形本身不是误报的理由——周期信号里的振幅/相位/基线变化同样可能是真实异常。

请按以下格式输出你的诊断报告。注意：必须给出你自己的判断和分析，不要照抄下面的格式说明。

## 判断结论
（从以下四个选项中选择一个，并说明依据：真实物理异常、传感器或采集异常、模型误报、数据不足无法判定）

## 依据分析
（对比异常分数与正常基线、观察波形形态变化、综合各层证据，写你自己的观察，两三句话）

## 置信度
（从高、中、低中选择一个，并说明理由）

最后另起一行，从以下三个选项中选择一个输出你的最终判断：
VERDICT: real
VERDICT: false_alarm
VERDICT: uncertain

用中文，简洁，只基于数据，不编造。总长不超过 300 字。"""


_VISION_SYSTEM_PROMPT = """你是航天器有效载荷健康管理（PHM）分析师。你将收到一张告警时刻的波形图和配套的统计数据。图中上方是遥测原始值波形（512点），下方是异常分数波形，红色虚线是异常阈值。

请像人类专家看监控图一样判断这条告警是真实异常还是误报。

判定依据：
1. 异常分数是否显著超过阈值（不只是擦线）？
2. 原始值波形是否有突变、漂移、振幅异常等真实异常特征？
3. 周期性波形里的振幅/基线变化也可能是真异常，不要仅因"看起来周期性"就判误报。
4. 结合文本中的正常基线对比（异常分数相对正常段 p95 的偏离倍数）。

请按以下格式输出你的诊断报告。注意：必须给出你自己的判断和分析，不要照抄下面的格式说明。

## 判断结论
（从以下四个选项中选择一个，并说明依据：真实物理异常、传感器或采集异常、模型误报、数据不足）

## 依据分析
（从波形形态和统计数据两方面分析，写你自己的观察，两三句话）

## 置信度
（从高、中、低中选择一个，并说明理由）

最后另起一行，从以下三个选项中选择一个输出你的最终判断：
VERDICT: real
VERDICT: false_alarm
VERDICT: uncertain

用中文，简洁，只基于数据，不编造。总长不超过 300 字。"""


_JOINT_SYSTEM_PROMPT = """你是航天器有效载荷健康管理（PHM）分析师。你正在分析一条**联合告警**——同一子系统（设备树 folder）下多个通道同时超过异常阈值时触发的子系统级告警。

重要：联合告警没有单通道原始波形数据，只有联合分数曲线和各子通道的异常分数贡献。不要假设存在波形图。

判定依据：
1. 联合分数是否显著超过阈值？联合分数曲线是否有明显的持续高峰（而非瞬时尖刺）？
2. 共识通道数：同时超阈值的通道越多，越可能是真实系统性异常（而非单通道偶发误报）。
3. 各子通道的异常分数：是多个通道都明显超阈值，还是只有少数通道擦线？
4. 子系统历史告警频率：频繁告警可能暗示检测模型对该子系统数据存在系统性误报倾向。

请按以下格式输出你的诊断报告。注意：必须给出你自己的判断和分析，不要照抄下面的格式说明。

## 判断结论
（从以下四个选项中选择一个，并说明依据：真实系统性异常、采集/通信异常、模型误报、数据不足无法判定）

## 依据分析
（基于联合分数曲线、各子通道贡献、共识通道数，写你自己的观察，两三句话。不要提到"波形图"——联合告警没有波形。）

## 置信度
（从高、中、低中选择一个，并说明理由）

最后另起一行，从以下三个选项中选择一个输出你的最终判断：
VERDICT: real
VERDICT: false_alarm
VERDICT: uncertain

用中文，简洁，只基于数据，不编造。总长不超过 300 字。"""


class DiagnosisService:
    """On-demand LLM anomaly diagnosis.

    Args:
        base_url: OpenAI-compatible API base URL (e.g.
            ``https://api.deepseek.com/v1``).  Empty string ⇒ service
            disabled (``diagnose`` returns a 503-style error dict).
        api_key: API key for the provider.
        model: chat model name (e.g. ``deepseek-chat``, ``qwen-plus``).
        warning_service: used to read ``_latest_cascade[channel]``.
        sqlite_store: used to query recent telemetry + historical alerts.
        config_service: used to resolve device-tree position / display name.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        warning_service,
        sqlite_store,
        config_service,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.warning_service = warning_service
        self.sqlite = sqlite_store
        self.config_service = config_service
        self._auto_lock = threading.Lock()
        self._auto_status: dict = {"running": False, "done": 0, "total": 0, "errors": 0}

    @property
    def enabled(self) -> bool:
        """Whether the service has the credentials to call the LLM API."""
        return bool(self.base_url and self.api_key and self.model)

    @property
    def auto_status(self) -> dict:
        """Current auto-diagnosis progress (thread-safe copy)."""
        with self._auto_lock:
            return dict(self._auto_status)

    _VERDICT_RE = re.compile(r"VERDICT:\s*(real|false_alarm|uncertain)", re.IGNORECASE)

    @staticmethod
    def _parse_verdict(text: str) -> str | None:
        """Extract the structured verdict from an LLM diagnosis report.

        Looks for a line matching ``VERDICT: real|false_alarm|uncertain``
        (case-insensitive).  Returns the lowercase verdict or None if not
        found / invalid.
        """
        if not text:
            return None
        m = DiagnosisService._VERDICT_RE.search(text)
        if m is None:
            return None
        return m.group(1).lower()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def diagnose(self, channel: str, alert_type: str = "measured",
                 alert_ts: float | None = None,
                 force_refresh: bool = False) -> dict:
        """Produce a diagnosis report for one channel.

        Args:
            channel: telemetry channel name (e.g. ``"C-1"``).
            alert_type: ``"measured"`` (space-side score) or ``"predicted"``
                (ground-side cascade).  Determines which context to emphasise.
            alert_ts: the alert/warning timestamp — used as the cache key
                so repeated clicks on the same alert return the stored
                diagnosis without re-calling the LLM.
            force_refresh: if True, bypass the cache and re-run the LLM.
                Used after prompt/data changes so stale diagnoses are
                regenerated with the latest code.

        Returns:
            ``{"channel", "alert_type", "diagnosis", "context_summary",
              "elapsed_sec", "error", "cached"}``.  On failure
            ``diagnosis`` is empty and ``error`` carries the reason.
        """
        # Cache lookup — one diagnosis per unique (channel, type, alert_ts).
        if not force_refresh and alert_ts is not None and self.sqlite is not None:
            cached = self.sqlite.get_diagnosis(channel, alert_type, alert_ts)
            if cached is not None and cached.get("diagnosis"):
                return {
                    "channel": channel,
                    "alert_type": alert_type,
                    "diagnosis": cached["diagnosis"],
                    "context_summary": cached.get("context_summary", {}),
                    "elapsed_sec": cached.get("elapsed_sec", 0.0),
                    "error": cached.get("error"),
                    "llm_verdict": cached.get("llm_verdict"),
                    "cached": True,
                }

        t0 = time.time()
        if not self.enabled:
            return {
                "channel": channel,
                "alert_type": alert_type,
                "diagnosis": "",
                "context_summary": {},
                "elapsed_sec": 0.0,
                "error": "LLM diagnosis not configured — set OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL",
                "llm_verdict": None,
                "cached": False,
            }

        context = self._build_context(channel, alert_type, alert_ts)
        if context is None:
            return {
                "channel": channel,
                "alert_type": alert_type,
                "diagnosis": "",
                "context_summary": {},
                "elapsed_sec": time.time() - t0,
                "error": f"no detection data available for channel {channel}",
                "llm_verdict": None,
                "cached": False,
            }

        user_prompt = self._build_prompt(channel, context)
        # Render a PNG waveform chart from the alert-time snapshot for the
        # vision LLM.  The chart + text prompt together give the LLM both
        # visual waveform shape and quantitative statistics.
        raw_snap = context.get("snapshot_raw")
        score_snap = context.get("snapshot_scores")
        image_b64 = None
        if raw_snap:
            try:
                threshold = context.get("summary", {}).get("threshold", 0.5)
                image_b64 = self._render_chart_b64(
                    raw_snap, score_snap or [], channel, threshold,
                )
            except Exception:
                logger.debug("chart render failed for %s", channel, exc_info=True)
        text, err = self._call_llm(
            user_prompt, image_b64=image_b64,
            system_prompt=_JOINT_SYSTEM_PROMPT if alert_type == "joint" else None,
        )
        elapsed = round(time.time() - t0, 2)
        verdict = self._parse_verdict(text) if text else None
        result = {
            "channel": channel,
            "alert_type": alert_type,
            "diagnosis": text,
            "context_summary": context["summary"],
            "elapsed_sec": elapsed,
            "error": err,
            "llm_verdict": verdict,
            "cached": False,
        }
        # Persist to DB for cache reuse (even on partial failure — the error
        # message is cached too so repeated clicks don't hammer the API).
        if alert_ts is not None and self.sqlite is not None:
            self.sqlite.save_diagnosis(
                channel, alert_type, alert_ts,
                diagnosis=text,
                context_summary=context["summary"],
                elapsed_sec=elapsed,
                error=err,
                verdict=verdict,
            )
        # Write verdict back to the alert/warning entry.
        if verdict is not None and alert_ts is not None:
            self._write_verdict_back(channel, alert_type, alert_ts, verdict)
        return result

    def _write_verdict_back(self, channel: str, alert_type: str,
                            alert_ts: float, verdict: str) -> None:
        """Write the LLM verdict back to the WarningEntry (predicted) or
        alert_records row (measured) so it appears in list endpoints."""
        try:
            if alert_type == "predicted":
                # Find the in-memory WarningEntry by (channel, created_at).
                ws = self.warning_service
                if ws and hasattr(ws, "warnings"):
                    store = ws.warnings
                    if hasattr(store, "all"):
                        for w in store.all():
                            if w.get("channel") == channel and w.get("created_at") == alert_ts:
                                wid = w.get("id", 0)
                                if wid and hasattr(store, "set_verdict"):
                                    store.set_verdict(wid, "llm", verdict)
                                break
            elif alert_type in ("measured", "joint") and self.sqlite is not None:
                # joint 告警也在 alert_records，复用 measured 的回写路径
                self.sqlite.update_alert_verdict(channel, alert_ts, verdict, is_llm=True)
        except Exception:
            logger.debug("write_verdict_back failed for %s/%s", channel, alert_type, exc_info=True)

    # ------------------------------------------------------------------
    # Auto-diagnosis mode (batch)
    # ------------------------------------------------------------------

    def auto_diagnose_all(self) -> dict:
        """Trigger batch diagnosis for all warnings + alerts without llm_verdict.

        Runs in a background daemon thread.  Progress is tracked in
        ``self._auto_status`` (exposed via the ``auto_status`` property).

        Returns ``{"started": True, "total": N}`` on success, or
        ``{"started": False, "error": "..."}`` if disabled or already running.
        """
        if not self.enabled:
            return {"started": False, "error": "LLM diagnosis not configured"}
        with self._auto_lock:
            if self._auto_status["running"]:
                return {"started": False, "error": "already running"}
            # Collect all targets that don't yet have an llm_verdict.
            targets: list[tuple[str, str, float]] = []
            # Predicted warnings (in-memory WarningStore).
            ws = self.warning_service
            if ws and hasattr(ws, "warnings"):
                store = ws.warnings
                if hasattr(store, "all"):
                    for w in store.all():
                        if w.get("llm_verdict") is None:
                            targets.append((w["channel"], "predicted", w["created_at"]))
            # Measured alerts (SQLite).
            if self.sqlite is not None:
                for a in self.sqlite.query_alerts(limit=500):
                    if a.get("llm_verdict") is None:
                        targets.append((a["channel"], "measured", a["created_at"]))
            self._auto_status = {
                "running": True, "done": 0,
                "total": len(targets), "errors": 0,
            }
        if not targets:
            with self._auto_lock:
                self._auto_status["running"] = False
            return {"started": True, "total": 0}
        threading.Thread(target=self._auto_run, args=(targets,), daemon=True).start()
        return {"started": True, "total": len(targets)}

    def _auto_run(self, targets: list[tuple[str, str, float]]) -> None:
        """Background worker: diagnose each target sequentially."""
        for channel, alert_type, alert_ts in targets:
            try:
                self.diagnose(channel, alert_type, alert_ts)
            except Exception:
                logger.debug("auto-diagnose failed for %s/%s", channel, alert_type, exc_info=True)
                with self._auto_lock:
                    self._auto_status["errors"] += 1
            with self._auto_lock:
                self._auto_status["done"] += 1
        with self._auto_lock:
            self._auto_status["running"] = False

    # ------------------------------------------------------------------
    # Context aggregation
    # ------------------------------------------------------------------

    def _build_context(self, channel: str, alert_type: str,
                       alert_ts: float | None = None) -> dict | None:
        """Gather all data points the diagnosis prompt needs.

        Returns ``None`` if there is no detection data at all for the
        channel (the LLM would have nothing to analyse).

        If *alert_ts* is provided, the method tries to read the alert-time
        **snapshot** (raw/score waveform captured when the alert fired)
        instead of the current telemetry window.  This is critical for
        late diagnosis: by the time a user clicks "diagnose", the
        telemetry table and in-memory ``_latest_cascade`` may have
        scrolled far past the alert point.
        """
        # --- Snapshot lookup (preferred over live telemetry) ---
        snapshot_raw: list | None = None
        snapshot_scores: list | None = None
        snapshot_pred: list | None = None
        snapshot_source: str = "none"

        if alert_ts is not None:
            if alert_type == "measured" and self.sqlite is not None:
                # Find the alert record by (channel, created_at≈alert_ts).
                for a in (self.sqlite.query_alerts(limit=200) or []):
                    if a.get("channel") == channel and abs(
                        (a.get("created_at") or 0) - alert_ts
                    ) < 1.0:
                        snapshot_raw = a.get("raw_snapshot")
                        snapshot_scores = a.get("score_snapshot")
                        if snapshot_raw is not None:
                            snapshot_source = "alert_records"
                        break
            elif alert_type == "predicted":
                # Find the WarningEntry by matching created_at.
                ws = self.warning_service.warnings
                for w in (ws.all() if hasattr(ws, "all") else []):
                    w_dict = w if isinstance(w, dict) else (w.to_dict() if hasattr(w, "to_dict") else {})
                    if w_dict.get("channel") == channel and abs(
                        (w_dict.get("created_at") or 0) - alert_ts
                    ) < 1.0:
                        snapshot_raw = w_dict.get("raw_snapshot")
                        snapshot_pred = w_dict.get("pred_snapshot")
                        snapshot_scores = w_dict.get("score_snapshot")
                        if snapshot_raw is not None:
                            snapshot_source = "warning_entry"
                        break

        # Latest cascade (three-layer detail with per-point scores).
        cascade: CascadeOutput | None = self.warning_service._latest_cascade.get(channel)
        cascade_dict = cascade.to_dict(max_detail=False) if cascade else None

        # Recent telemetry (raw + predicted values for trend).
        # query_window returns a dict with a "data" key (list of row dicts).
        recent_rows = []
        try:
            if self.sqlite is not None:
                win = self.sqlite.query_window(channel, count=512)
                recent_rows = (win or {}).get("data", []) if isinstance(win, dict) else []
        except Exception:
            logger.debug("query_window failed for %s", channel, exc_info=True)

        # Historical alerts for recurrence context.  query_alerts has no
        # channel filter, so we fetch the last N and filter in-memory.
        history = []
        try:
            if self.sqlite is not None:
                all_alerts = self.sqlite.query_alerts(limit=50)
                history = [a for a in (all_alerts or []) if a.get("channel") == channel][:10]
        except Exception:
            logger.debug("query_alerts failed for %s", channel, exc_info=True)

        # Predict scores (forecast-segment anomaly scores).
        predict_scores = self.warning_service._latest_predict_scores.get(channel)

        # Device-tree position + display name.
        display_name, device_path, description = self._resolve_device(channel)

        # ── Joint alert（子系统联合告警）专用分支 ──────────────────────
        # joint 告警的 channel 是虚拟 "SUB:<folder>"，无遥测表/cascade/raw_snapshot。
        # 它的上下文全部存在 alert_records.score_snapshot 里（dict，含 joint_curve /
        # contributions / channels）。这里提取后直接构造 context 提前返回，
        # 不走下面的 measured/predicted 通用逻辑（那套依赖遥测表，对 joint 全空）。
        if alert_type == "joint":
            return self._build_joint_context(
                channel, alert_ts, display_name, device_path, description, history,
            )

        if cascade_dict is None and not recent_rows and snapshot_raw is None:
            return None

        # Window stats: prefer the alert-time snapshot if available (it
        # captures the exact waveform that triggered the alert).  Fall back
        # to the current telemetry window for old data without snapshots.
        if snapshot_raw is not None:
            window_stats = self._summarise_snapshot(
                snapshot_raw, snapshot_pred, snapshot_scores,
            )
            if snapshot_source != "none":
                logger.debug("diagnosis using %s snapshot for %s", snapshot_source, channel)
        else:
            window_stats = self._summarise_window(recent_rows)

        # Normal-baseline contrast: from a wider history window, extract
        # rows whose anomaly_score is below threshold (i.e. "normal" samples)
        # and compute their mean/std.  This gives the LLM a reference for
        # "how much does the current value deviate vs. normal fluctuation",
        # countering the prior that any alert must be real.
        baseline_stats = self._compute_baseline(channel)

        # Cascade layer summary (decisions + scores + rules).
        layer_summary = self._summarise_cascade(cascade_dict)

        # For measured alerts, the ground-side cascade reflects the
        # ground's PREDICTION-segment evaluation — a different data slice
        # than what the space segment actually flagged.  This mismatch
        # causes the LLM to see a final_score=0 (ground eval found nothing)
        # while the alert was triggered by the space-segment TSPulse.
        # When we have the alert-time snapshot scores (the authoritative
        # detection result), prefer them over the mismatched ground cascade.
        if alert_type == "measured" and snapshot_scores:
            scores_arr = np.array(snapshot_scores, dtype=np.float64)
            snapshot_l2 = float(scores_arr.max())
            layer_summary = {
                "final_score": snapshot_l2,
                "l1_decision": layer_summary.get("l1_decision") if layer_summary else None,
                "l1_score": layer_summary.get("l1_score", 0.0) if layer_summary else 0.0,
                "l2_score": snapshot_l2,
                "l3_score": layer_summary.get("l3_score", 0.0) if layer_summary else 0.0,
                "l3_rules": layer_summary.get("l3_rules", []) if layer_summary else [],
                "flip": layer_summary.get("flip") if layer_summary else None,
                "_source": "snapshot",
            }
        elif layer_summary is None and snapshot_scores:
            # Fallback for any alert type when cascade is entirely missing.
            scores_arr = np.array(snapshot_scores, dtype=np.float64)
            layer_summary = {
                "final_score": float(scores_arr.max()),
                "l1_decision": None,
                "l1_score": 0.0,
                "l2_score": float(scores_arr.max()),
                "l3_score": 0.0,
                "l3_rules": [],
                "flip": None,
                "_source": "snapshot",
            }

        summary = {
            "channel": channel,
            "display_name": display_name,
            "description": description,
            "device_path": device_path,
            "alert_type": alert_type,
            "threshold": ANOMALY_THRESHOLD,
            "n_recent_points": len(recent_rows),
            "window_stats": window_stats,
            "baseline": baseline_stats,
            "cascade": layer_summary,
            "n_history_alerts": len(history),
            "history_statuses": [h.get("status", "?") for h in history][:5],
            "predict_horizon": FORECAST_PREDICTION_LENGTH,
            "predict_max_score": (
                float(np.max(predict_scores["scores"]))
                if predict_scores and predict_scores.get("scores")
                else None
            ),
        }
        return {
            "summary": summary,
            "cascade_dict": cascade_dict,
            "window_stats": window_stats,
            "layer_summary": layer_summary,
            "display_name": display_name,
            "device_path": device_path,
            "snapshot_raw": snapshot_raw,
            "snapshot_scores": snapshot_scores,
        }

    def _build_joint_context(
        self, channel: str, alert_ts: float | None,
        display_name: str | None, device_path: str | None,
        description: str | None, history: list,
    ) -> dict | None:
        """构造联合告警（joint）的诊断上下文。

        joint 告警的 channel 是虚拟 ``SUB:<folder>``，无遥测表/cascade。
        上下文来自 alert_records.score_snapshot（dict，含 joint_curve /
        contributions / channels）。找不到对应告警记录时返回 None。
        """
        if self.sqlite is None or alert_ts is None:
            return None
        # 从 alert_records 按 (channel, alert_ts) 找到这条联合告警
        joint_alert = None
        try:
            for a in (self.sqlite.query_alerts(limit=200) or []):
                if a.get("channel") == channel and abs(
                    (a.get("created_at") or 0) - alert_ts
                ) < 1.0:
                    joint_alert = a
                    break
        except Exception:
            logger.debug("joint alert lookup failed for %s", channel, exc_info=True)
        if not joint_alert:
            return None

        snap = joint_alert.get("score_snapshot")
        if not isinstance(snap, dict):
            snap = {}
        joint_curve = snap.get("joint_curve") or []
        contributions = snap.get("contributions") or []
        sub_channels = snap.get("channels") or []
        joint_score = joint_alert.get("score") or 0.0

        summary = {
            "channel": channel,
            "display_name": display_name or channel,
            "description": description,
            "device_path": device_path,
            "alert_type": "joint",
            "threshold": ANOMALY_THRESHOLD,
            "joint_score": float(joint_score),
            "joint_curve": joint_curve,
            "contributions": contributions,
            "sub_channels": sub_channels,
            "n_history_alerts": len(history),
            "history_statuses": [h.get("status", "?") for h in history][:5],
            "predict_horizon": FORECAST_PREDICTION_LENGTH,
            "predict_max_score": None,
            "n_recent_points": 0,
            "window_stats": None,
            "baseline": None,
            "cascade": None,
        }
        return {
            "summary": summary,
            "cascade_dict": None,
            "window_stats": None,
            "layer_summary": None,
            "display_name": display_name,
            "device_path": device_path,
            "snapshot_raw": None,
            "snapshot_scores": None,
            "_joint": True,  # 标记，供 _build_prompt 识别走 joint 模板
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, channel: str, context: dict) -> str:
        s = context["summary"]
        # 联合告警走专用 prompt 模板（多通道共识，非单通道波形）
        if context.get("_joint") or s.get("alert_type") == "joint":
            return self._build_joint_prompt(channel, s)
        ws = s["window_stats"]
        ls = s["cascade"]
        lines = [
            f"传感器：{s['display_name'] or channel}",
        ]
        # Sensor description — the most important context for a human analyst.
        # Without knowing what physical quantity this sensor measures, the LLM
        # cannot judge whether a deviation is meaningful (e.g. a 0.1°C drift on
        # a temperature sensor vs a 0.1A spike on a current sensor).
        desc = s.get("description")
        if desc:
            lines.append(f"传感器描述：{desc}")
        if s["device_path"]:
            lines.append(f"设备位置：{s['device_path']}")
        lines += [
            f"数据来源：{'实测段（space端TSPulse检测）' if s['alert_type']=='measured' else '预测段（ground端级联检测）'}",
            f"异常阈值：{s['threshold']}",
            "",
            f"【告警时刻波形】（附图：上方=原始值，下方=异常分数，红色虚线=阈值）",
        ]
        if ws:
            lines += [
                f"- 原始值：mean={ws['raw_mean']:.4f}, std={ws['raw_std']:.4f}, "
                f"min={ws['raw_min']:.4f}, max={ws['raw_max']:.4f}",
                f"- 最新原始值：{ws['raw_last']:.4f}",
                f"- 偏离均值：{ws['raw_last_sigma']:.1f}σ",
            ]
            if ws.get("pred_last") is not None:
                lines.append(f"- 预测值（最近）：{ws['pred_last']:.4f}")
        else:
            lines.append("- （无遥测数据）")

        # Waveform shape — the KEY section that lets the LLM "see" what
        # a human sees on a chart.  Without this, mean/std make a periodic
        # sine and a mid-window step jump look identical.
        wf = ws.get("waveform") if ws else None
        if wf:
            lines += [
                "",
                "【波形形态分析】（512 点降采样到 32 点，█=高 ▁=低，从左=早到右=晚）",
                f"- 异常分数峰值位置：{wf['score_peak_pos']}",
                f"- 异常宽度：{wf['score_width']}（超阈值占比 {wf['score_over_frac']:.1%}）",
                f"- 异常分数最大值：{wf['score_max']:.4f}",
                f"- 原始值趋势：{wf['raw_trend']}",
                f"- 周期性：{wf['raw_periodicity']}",
                f"- 原始值波形：{wf.get('raw_sparkline', '')}",
                f"- 异常分数波形：{wf.get('score_sparkline', '')}",
                f"- 原始值数值（32点）：{wf['raw_downsample']}",
                f"- 异常分数数值（32点）：{wf['score_downsample']}",
            ]

        # Normal-baseline contrast — let the LLM judge whether the current
        # deviation is truly outside normal fluctuation.
        bl = s.get("baseline")
        if bl and ws:
            dev_sigma = abs(ws["raw_last"] - bl["mean"]) / bl["std"]
            lines += [
                "",
                f"【正常基线对比】（{bl['n']} 个正常样本，anomaly_score<{s['threshold']}）",
                f"- 正常段原始值：mean={bl['mean']:.4f}, std={bl['std']:.4f}, "
                f"range=[{bl['min']:.4f}, {bl['max']:.4f}]",
                f"- 当前值相对正常基线偏离：{dev_sigma:.1f}σ"
                + ("（超出正常波动范围）" if dev_sigma > 3 else "（在正常波动范围内）"),
            ]
            # Score-based baseline comparison — the most informative signal
            # for distinguishing real anomalies from false alarms.  Compares
            # the alert-time peak score against what the detection model
            # normally outputs on this channel's normal data.
            if "score_mean" in bl and wf:
                alert_score = wf.get("score_max", 0)
                score_std = bl["score_std"] if bl["score_std"] > 1e-9 else 1e-9
                alert_sigma = (alert_score - bl["score_mean"]) / score_std
                lines += [
                    f"- 正常段异常分数：mean={bl['score_mean']:.4f}, "
                    f"std={bl['score_std']:.4f}, p95={bl['score_p95']:.4f}",
                    f"- 告警时刻异常分数峰值：{alert_score:.4f}"
                    f"（是正常段均值的 {alert_sigma:.1f}σ，"
                    f"{'超出' if alert_score > bl['score_p95'] else '未超'} 正常段 p95）",
                ]

        lines += ["", "【三层级联检测结果】"]
        if ls:
            src_tag = "（来源：告警时刻快照）" if ls.get("_source") == "snapshot" else ""
            lines += [
                f"- 最终异常分数：{ls['final_score']:.4f}"
                + (f"（L1={ls['l1_decision']}/{ls['l1_score']:.3f}）" if ls.get("l1_decision") else "")
                + src_tag,
                f"- L2 深度检测（TSPulse）：score={ls['l2_score']:.4f}"
                + (f"，方向翻转={'是' if ls.get('flip') else '否'}" if ls.get("flip") is not None else ""),
                f"- L3 物理约束：score={ls['l3_score']:.4f}"
                + (f"，触发规则={ls['l3_rules']}" if ls.get("l3_rules") else "无"),
            ]
        else:
            lines.append("- （无级联检测数据）")

        if s["predict_max_score"] is not None:
            ratio = s["predict_max_score"] / s["threshold"] if s["threshold"] else 0
            lines += [
                "",
                f"【未来 {s['predict_horizon']} 点预测】",
                f"- 预测段最大异常分数：{s['predict_max_score']:.4f}"
                + f"（阈值 {s['threshold']:.2f}，为阈值的 {ratio:.1f} 倍）",
                "- 注意：预测模型（TTM-R3）存在已知的振幅低估缺陷，"
                "低预测分数不代表异常会消失，仅作参考。",
            ]

        lines += [
            "",
            f"【历史告警】过去共 {s['n_history_alerts']} 条，最近状态：{s['history_statuses'] or '无'}",
            "",
            "请判断本条告警是真实异常还是误报，并给出依据。",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_joint_prompt(channel: str, s: dict) -> str:
        """联合告警专用 prompt（多通道共识，无单通道波形）。

        联合告警是同一子系统（设备树 folder）下 >=2 个 sibling 通道同时超阈值时
        触发的「子系统级」告警。prompt 聚焦：子系统名 + 联合分数 + 各子通道贡献 +
        共识通道数，让 LLM 判断是否为真实系统性异常。
        """
        sub_channels = s.get("sub_channels") or []
        contributions = s.get("contributions") or []
        joint_curve = s.get("joint_curve") or []
        joint_score = s.get("joint_score", 0.0)
        lines = [
            f"子系统联合告警：{s['display_name'] or channel}",
            f"告警类型：联合告警（{len(sub_channels)} 通道共识）",
            f"异常阈值：{s['threshold']}",
            "",
            f"【联合异常分数】{joint_score:.4f}"
            + (f"（{'超' if joint_score > s['threshold'] else '未超'}阈值）" if joint_score else ""),
        ]
        if s.get("device_path"):
            lines.append(f"设备位置：{s['device_path']}")
        if s.get("description"):
            lines.append(f"子系统描述：{s['description']}")

        # 各子通道贡献（contributions 是 list[dict]，含 channel/score/threshold/over 等）
        if contributions:
            lines += ["", f"【各子通道贡献】（共 {len(contributions)} 个通道）"]
            for c in contributions:
                if not isinstance(c, dict):
                    continue
                ch = c.get("channel", "?")
                sc = c.get("score", 0.0)
                th = c.get("threshold", s["threshold"])
                over = c.get("over", sc > th)
                lines.append(
                    f"- {ch}：异常分数 {sc:.4f}（阈值 {th:.2f}，"
                    f"{'超限' if over else '未超限'}）"
                )

        # 联合分数曲线（降采样到 ~16 点，避免 prompt 过长）
        if joint_curve and len(joint_curve) > 1:
            try:
                import numpy as _np
                arr = _np.array(joint_curve, dtype=float)
                n = len(arr)
                target = 16
                if n > target:
                    idx = _np.linspace(0, n - 1, target).astype(int)
                    arr = arr[idx]
                downsample = ",".join(f"{v:.3f}" for v in arr)
                peak = float(max(joint_curve))
                lines += [
                    "",
                    f"【联合分数曲线】（{len(joint_curve)} 点降采样到 {len(arr)} 点）",
                    f"- 峰值：{peak:.4f}",
                    f"- 曲线：{downsample}",
                ]
            except Exception:
                logger.debug("joint_curve downsample failed", exc_info=True)

        lines += [
            "",
            f"【历史告警】该子系统过去共 {s['n_history_alerts']} 条，"
            f"最近状态：{s['history_statuses'] or '无'}",
            "",
            "请判断本次联合告警是真实系统性异常还是误报，并给出依据。"
            "重点考虑：多通道同时超阈值是否反映真实物理异常，"
            "还是检测模型对该子系统数据的系统性误报。",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Waveform chart rendering for vision LLM input
    # ------------------------------------------------------------------

    @staticmethod
    def _render_chart_b64(
        raw: list[float], score: list[float], channel: str,
        threshold: float = 0.5,
    ) -> str | None:
        """Render a 2-panel waveform chart (raw + score) as base64 PNG.

        The chart gives a vision LLM the same visual information a human
        analyst sees on the monitoring dashboard: the raw telemetry
        waveform and the anomaly-score waveform with the detection
        threshold line, side by side over the same 512-sample window.

        Returns ``None`` if the input data is insufficient to render.
        """
        if not raw or len(raw) == 0:
            return None
        raw_arr = np.array(raw, dtype=np.float64)
        x = np.arange(len(raw_arr))
        score_arr = (
            np.array(score, dtype=np.float64) if score and len(score) > 0 else None
        )

        fig, axes = plt.subplots(
            2 if score_arr is not None else 1, 1,
            figsize=(8, 4), dpi=80, sharex=True,
        )
        if score_arr is None:
            axes = [axes]

        # Top panel: raw telemetry values.
        axes[0].plot(x, raw_arr, color="#2d8cf0", linewidth=0.8)
        axes[0].set_ylabel("Raw Value", fontsize=9)
        axes[0].set_title(f"Channel {channel} — Alert-time Waveform", fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=8)

        # Bottom panel: anomaly scores with threshold line.
        if score_arr is not None and len(axes) > 1:
            axes[1].plot(x, score_arr, color="#ff9900", linewidth=0.8)
            axes[1].axhline(
                y=threshold, color="red", linestyle="--", linewidth=1,
                label=f"threshold={threshold}",
            )
            axes[1].set_ylabel("Anomaly Score", fontsize=9)
            axes[1].set_xlabel("Sample (512 points)", fontsize=9)
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(labelsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # ------------------------------------------------------------------
    # LLM API call
    # ------------------------------------------------------------------

    def _call_llm(
        self, user_prompt: str, image_b64: str | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, str | None]:
        """Call the chat-completions endpoint. Returns (text, error).

        When ``image_b64`` is provided (a base64-encoded PNG), the request
        uses the vision-capable multimodal message format (``image_url`` +
        ``text`` content blocks).  This requires a vision model (e.g.
        GLM-4V-Flash).  When ``image_b64`` is ``None``, falls back to the
        plain text format for backward compatibility with text-only models.

        ``system_prompt`` overrides the default system prompt selection
        (used by joint alerts which need a waveform-free prompt).
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Build the user message: vision format (image + text) when a chart
        # is available, plain text otherwise.
        if image_b64:
            user_content = [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                }},
                {"type": "text", "text": user_prompt},
            ]
        else:
            user_content = user_prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or (_VISION_SYSTEM_PROMPT if image_b64 else _SYSTEM_PROMPT)},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 768,
        }
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return "", f"LLM API returned {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text.strip(), None
        except httpx.TimeoutException:
            return "", "LLM API timeout (45s)"
        except Exception as e:
            return "", f"LLM API error: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, channel: str) -> tuple[str | None, str | None, str | None]:
        """Look up display name, device-tree path, and description for a channel.

        Returns ``(display_name, device_path, description)``.  The description
        is the human-readable sensor info (what physical quantity, unit,
        normal range) that a human analyst needs to judge an alert — without
        it the LLM is diagnosing a meaningless channel code like "C-1".
        """
        try:
            if self.config_service is None:
                return None, None, None
            tree = self.config_service.load()
            if not tree:
                return None, None, None
            from ..services.tree_utils import get_flat_sensors

            sensors = get_flat_sensors(tree)
            path_parts = []
            display = None
            description = None
            # Build a channel → folder-path map by walking the tree.
            def _walk(node, prefix):
                nonlocal display, description
                if node.get("type") == "sensor":
                    sid = node.get("sourceId", "")
                    # Match by sourceId (e.g. "file:NASA-MSL/C-1") or
                    # channelName (e.g. "C-1").
                    if sid == channel or node.get("channelName") == channel:
                        display = node.get("name")
                        description = node.get("description")
                        path_parts.extend(prefix + [node.get("name", channel)])
                else:
                    children = node.get("children", [])
                    for c in children:
                        _walk(c, prefix + [node.get("name", "")])

            # tree is the full config dict: {"device_tree": [...], ...}
            # device_tree is a LIST of top-level nodes (folders/sensors).
            device_tree = tree.get("device_tree", []) if isinstance(tree, dict) else (tree if isinstance(tree, list) else [])
            for c in device_tree:
                _walk(c, [])
            if display is None:
                # Fallback: linear search in flat sensors.
                for sn in sensors:
                    if sn.get("sourceId") == channel or sn.get("channelName") == channel:
                        display = sn.get("name")
                        description = sn.get("description")
                        break
            return (
                display,
                " / ".join(p for p in path_parts if p) or None,
                description,
            )
        except Exception:
            logger.debug("device resolve failed for %s", channel, exc_info=True)
            return None, None, None

    @staticmethod
    def _summarise_window(rows: list) -> dict | None:
        """Compact statistics over recent telemetry rows."""
        if not rows:
            return None
        raws = np.array(
            [r.get("raw_value") for r in rows if r.get("raw_value") is not None],
            dtype=np.float64,
        )
        if len(raws) == 0:
            return None
        mu, sigma = float(raws.mean()), float(raws.std())
        sigma = sigma if sigma > 1e-9 else 1e-9
        preds = [r.get("predicted_value") for r in rows if r.get("predicted_value") is not None]
        return {
            "raw_mean": mu,
            "raw_std": sigma,
            "raw_min": float(raws.min()),
            "raw_max": float(raws.max()),
            "raw_last": float(raws[-1]),
            "raw_last_sigma": abs(float(raws[-1]) - mu) / sigma,
            "pred_last": float(preds[-1]) if preds else None,
            "n": len(raws),
        }

    @staticmethod
    def _summarise_snapshot(
        raw_vals: list, pred_vals: list | None = None,
        score_vals: list | None = None,
    ) -> dict | None:
        """Compact statistics over an alert-time snapshot (raw waveform).

        Unlike ``_summarise_window`` which takes telemetry row dicts,
        this takes the raw list of floats captured at alert time — so
        it reflects the exact waveform that triggered the alert, not
        the current telemetry window which may have scrolled past.

        When ``score_vals`` is provided (per-sample anomaly scores from
        the alert-time snapshot), waveform shape features are computed
        so the LLM can "see" the anomaly's temporal structure instead
        of only scalar aggregates.
        """
        if not raw_vals:
            return None
        raws = np.array(raw_vals, dtype=np.float64)
        if len(raws) == 0:
            return None
        mu, sigma = float(raws.mean()), float(raws.std())
        sigma = sigma if sigma > 1e-9 else 1e-9
        pred_last = None
        if pred_vals and len(pred_vals) > 0:
            pred_last = float(pred_vals[-1])
        result = {
            "raw_mean": mu,
            "raw_std": sigma,
            "raw_min": float(raws.min()),
            "raw_max": float(raws.max()),
            "raw_last": float(raws[-1]),
            "raw_last_sigma": abs(float(raws[-1]) - mu) / sigma,
            "pred_last": pred_last,
            "n": len(raws),
        }
        # Waveform shape features — give the LLM temporal structure that
        # mean/std completely destroy.  Without this, a periodic sine and
        # a mid-window step jump look identical to the LLM.
        if score_vals and len(score_vals) > 0:
            scores_arr = np.array(score_vals, dtype=np.float64)
            wf = DiagnosisService._extract_waveform_features(raws, scores_arr)
            result["waveform"] = wf
        return result

    # ── Waveform shape analysis for the LLM ───────────────────────────

    @staticmethod
    def _downsample_curve(values: np.ndarray, n_bins: int = 32) -> list[float]:
        """Downsample a 1-D curve to ``n_bins`` representative points.

        Splits the array into ``n_bins`` equal-length segments and takes
        each segment's mean.  This preserves the temporal structure (peak
        position, trend) while reducing 512 points to ~32 numbers that
        fit comfortably in an LLM prompt (~100 tokens).
        """
        if len(values) == 0:
            return []
        n = len(values)
        if n <= n_bins:
            return [float(v) for v in values]
        # np.array_split handles uneven division gracefully.
        segments = np.array_split(values, n_bins)
        return [float(seg.mean()) for seg in segments]

    @staticmethod
    def _extract_waveform_features(
        raws: np.ndarray, scores: np.ndarray,
    ) -> dict:
        """Extract human-readable waveform shape descriptors.

        These let a non-vision LLM "see" what a human sees on a chart:
        where the anomaly is, how wide it is, whether the signal is
        trending or periodic.  All features are categorical or ratios,
        not raw arrays, to minimise token cost.
        """
        n = len(scores)
        threshold = float(ANOMALY_THRESHOLD)

        # --- Score-curve features ---
        # Peak position: where in the window is the max score?
        peak_idx = int(np.argmax(scores))
        if peak_idx < n / 3:
            peak_pos = "前段（1/3）"
        elif peak_idx < 2 * n / 3:
            peak_pos = "中段（1/3）"
        else:
            peak_pos = "后段（1/3）"

        # Anomaly width: fraction of points above threshold
        over_mask = scores > threshold
        over_frac = float(over_mask.sum()) / n if n > 0 else 0.0
        if over_frac == 0:
            width_desc = "无超阈值点"
        elif over_frac < 0.1:
            width_desc = "窄峰（<10% 超阈值）"
        elif over_frac < 0.3:
            width_desc = "中等（10-30% 超阈值）"
        else:
            width_desc = "宽峰（>30% 超阈值）"

        # Score curve downsample (32 points) for the LLM to "see" shape
        score_downsample = DiagnosisService._downsample_curve(scores, 32)

        # --- Raw-curve features ---
        raw_downsample = DiagnosisService._downsample_curve(raws, 32)

        # Trend: compare first-third vs last-third mean
        third = max(1, n // 3)
        first_mean = float(raws[:third].mean())
        last_mean = float(raws[-third:].mean())
        raw_std = float(raws.std())
        if raw_std < 1e-9:
            trend = "平稳（近常数）"
        elif abs(last_mean - first_mean) < 0.5 * raw_std:
            trend = "平稳（无趋势）"
        elif last_mean > first_mean:
            trend = "上升趋势"
        else:
            trend = "下降趋势"

        # Periodicity: autocorrelation at lag 1.
        # Periodic signals have high positive autocorrelation; step changes
        # and noise don't.  We guard against near-constant segments (which
        # can inflate autocorrelation artificially) by checking local
        # variation: compute std of the first and last thirds separately.
        if raw_std > 1e-9 and n > 4:
            normalized = (raws - raws.mean()) / raw_std
            ac1 = float(np.mean(normalized[:-1] * normalized[1:]))
            # Detect step-change: if the signal has a large DC shift between
            # first and last third, it's a step/trend, not periodic — even
            # if autocorrelation is high (constant segments inflate it).
            # For a clean step, |diff|/std ≈ 2.0; for a periodic sine, < 0.5.
            is_step = abs(last_mean - first_mean) >= 1.5 * raw_std
            if is_step:
                periodicity = "非周期（阶跃跳变）"
            elif ac1 > 0.7:
                periodicity = "强周期性（类似正弦）"
            elif ac1 > 0.3:
                periodicity = "弱周期性"
            else:
                periodicity = "非周期（突变/噪声）"
        else:
            periodicity = "无法判定（近常数）"

        return {
            "score_peak_pos": peak_pos,
            "score_width": width_desc,
            "score_over_frac": round(over_frac, 3),
            "score_max": float(np.max(scores)),
            "raw_trend": trend,
            "raw_periodicity": periodicity,
            "raw_downsample": [round(v, 4) for v in raw_downsample],
            "score_downsample": [round(v, 4) for v in score_downsample],
            "raw_sparkline": DiagnosisService._to_sparkline(raw_downsample),
            "score_sparkline": DiagnosisService._to_sparkline(score_downsample),
        }

    # ── ASCII sparkline rendering ──────────────────────────────────────

    _SPARK_CHARS = "▁▂▃▄▅▆▇█"

    @staticmethod
    def _to_sparkline(values: list[float]) -> str:
        """Render a numeric array as a compact Unicode sparkline string.

        Uses 8 block characters (▁..█) to map the value range, producing a
        ~32-character single-line "mini chart" that a text LLM can visually
        parse to recognise waveform shape (peaks, troughs, trends, periodicity)
        — far more effectively than a flat list of numbers.
        """
        if not values:
            return ""
        arr = np.array(values, dtype=np.float64)
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax - vmin < 1e-9:
            # Near-constant: render as mid-level line.
            return DiagnosisService._SPARK_CHARS[3] * len(values)
        # Normalise to [0, 7] and map to spark chars.
        norm = (arr - vmin) / (vmax - vmin) * (len(DiagnosisService._SPARK_CHARS) - 1)
        indices = np.clip(np.round(norm).astype(int), 0, len(DiagnosisService._SPARK_CHARS) - 1)
        return "".join(DiagnosisService._SPARK_CHARS[i] for i in indices)

    def _compute_baseline(self, channel: str) -> dict | None:
        """Normal-segment baseline for a channel.

        Queries a wider history window (2048 points) and extracts rows
        whose ``anomaly_score`` is below the detection threshold (i.e.
        "normal" samples).  Returns their mean/std so the LLM can compare
        the current deviation against normal fluctuation.

        Returns ``None`` if there are insufficient normal samples.
        """
        if self.sqlite is None:
            return None
        try:
            win = self.sqlite.query_window(channel, count=2048)
            rows = (win or {}).get("data", []) if isinstance(win, dict) else []
        except Exception:
            logger.debug("baseline query_window failed for %s", channel, exc_info=True)
            return None
        normal = [
            r for r in rows
            if r.get("raw_value") is not None
            and (r.get("anomaly_score") or 0) < ANOMALY_THRESHOLD
        ]
        if len(normal) < 30:
            return None
        raws = np.array([r["raw_value"] for r in normal], dtype=np.float64)
        mu, sigma = float(raws.mean()), float(raws.std())
        sigma = sigma if sigma > 1e-9 else 1e-9
        # Also compute normal-segment anomaly-score stats — this lets the
        # LLM compare the alert-time score against what the model normally
        # outputs on this channel, which is far more informative than
        # comparing against a fixed threshold alone.
        normal_scores = np.array(
            [r["anomaly_score"] for r in normal if r.get("anomaly_score") is not None],
            dtype=np.float64,
        )
        result = {
            "mean": mu,
            "std": sigma,
            "min": float(raws.min()),
            "max": float(raws.max()),
            "n": len(normal),
        }
        if len(normal_scores) > 0:
            result["score_mean"] = float(normal_scores.mean())
            result["score_std"] = float(normal_scores.std())
            result["score_p95"] = float(np.percentile(normal_scores, 95))
        return result

    @staticmethod
    def _summarise_cascade(cascade_dict: dict | None) -> dict | None:
        """Extract layer decisions/scores/rules from a CascadeOutput dict."""
        if not cascade_dict:
            return None
        layers = cascade_dict.get("layers", [])
        out = {
            "final_score": float(cascade_dict.get("final_score", 0.0) or 0.0),
            "l1_decision": None,
            "l1_score": 0.0,
            "l2_score": 0.0,
            "l3_score": 0.0,
            "l3_rules": [],
            "flip": None,
        }
        for layer in layers:
            name = layer.get("layer", "")
            if name.endswith("L1_classic") or name == "L1_classic":
                out["l1_decision"] = layer.get("decision")
                out["l1_score"] = float(layer.get("score", 0.0) or 0.0)
                detail = layer.get("detail", {})
                if isinstance(detail, dict) and detail.get("rules"):
                    out["l1_rules"] = detail["rules"]
            elif name.endswith("L2_dl") or name == "L2_dl":
                out["l2_score"] = float(layer.get("score", 0.0) or 0.0)
                detail = layer.get("detail", {})
                if isinstance(detail, dict):
                    out["flip"] = detail.get("flip")
            elif name.endswith("L3_physical") or name == "L3_physical":
                out["l3_score"] = float(layer.get("score", 0.0) or 0.0)
                detail = layer.get("detail", {})
                if isinstance(detail, dict) and detail.get("rules"):
                    out["l3_rules"] = detail["rules"]
        return out
