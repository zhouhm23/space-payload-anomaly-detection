"""LLM-powered anomaly diagnosis service.

Aggregates the per-channel detection context (three-layer cascade output,
recent telemetry statistics, historical alerts, device-tree position,
offline calibration) into a structured prompt, calls an OpenAI-compatible
chat-completions API, and returns a Markdown diagnosis report covering
root-cause analysis and trend assessment.

The service is **on-demand** — it only runs when a user requests a
diagnosis for a specific alert/warning.  Diagnosis results are cached
in SQLite (``diagnosis_records`` table) keyed by ``(channel, alert_type,
alert_ts)`` so repeated clicks return the stored report without
re-calling the LLM.  A structured ``llm_verdict`` (real / false_alarm /
uncertain) is parsed from the report and written back to the
WarningEntry (predicted) or alert_records row (measured).

Configuration is via standard environment variables so any
OpenAI-compatible provider works::

    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://api.deepseek.com/v1   # or any compatible endpoint
    LLM_MODEL=deepseek-chat                        # default chat model
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

import httpx
import numpy as np

from ..config import (
    ANOMALY_THRESHOLD,
    FORECAST_PREDICTION_LENGTH,
)
from ..algorithm.cascade_types import CascadeOutput

logger = logging.getLogger(__name__)

__all__ = ["DiagnosisService"]


_SYSTEM_PROMPT = """你是航天器有效载荷健康管理（PHM）分析师。你的任务是判断一条告警是真实异常还是误报。

重要：告警已触发不代表一定是真实异常。检测模型（TSPulse 重建误差）会产生误报，尤其在周期性波形、数据漂移、短窗口等情况下。你需要基于数据独立判断，不要预设告警为真。

报告格式（Markdown），不要复述输入数据：

## 判断结论
先给出你的判断，四选一：真实物理异常 / 传感器或采集异常 / 模型误报 / 数据不足无法判定。一句话结论 + 关键依据。

## 依据分析
对比当前数值与正常基线（如有）：偏离幅度是否足够大、是否超出正常波动范围、级联各层分数是否一致支持异常。两三句话。

## 置信度
高/中/低 + 一句话理由。

最后另起一行输出结构化判断，三选一：
VERDICT: real          （判定为真实异常）
VERDICT: false_alarm   （判定为误报）
VERDICT: uncertain     （无法确定）

判定原则：只有当偏离幅度显著超出正常波动、且多层检测一致支持时才判 real；若偏离在正常范围内或仅单层触发，倾向于 false_alarm 或 uncertain。用中文，简洁，只基于数据，不编造。总长不超过 300 字。"""


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
        text, err = self._call_llm(user_prompt)
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
            elif alert_type == "measured" and self.sqlite is not None:
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
        display_name, device_path = self._resolve_device(channel)

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

        summary = {
            "channel": channel,
            "display_name": display_name,
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
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, channel: str, context: dict) -> str:
        s = context["summary"]
        ws = s["window_stats"]
        ls = s["cascade"]
        lines = [
            f"通道：{channel}" + (f"（{s['display_name']}）" if s["display_name"] else ""),
            f"设备位置：{s['device_path'] or '未知'}",
            f"数据来源：{'实测段（space端TSPulse检测）' if s['alert_type']=='measured' else '预测段（ground端级联检测）'}",
            f"异常阈值：{s['threshold']}",
            "",
            f"【最近 {s['n_recent_points']} 点遥测统计】",
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
                "【波形形态分析】（降采样到 32 点，保留时间结构）",
                f"- 异常分数峰值位置：{wf['score_peak_pos']}",
                f"- 异常宽度：{wf['score_width']}（超阈值占比 {wf['score_over_frac']:.1%}）",
                f"- 异常分数最大值：{wf['score_max']:.4f}",
                f"- 原始值趋势：{wf['raw_trend']}",
                f"- 周期性：{wf['raw_periodicity']}",
                f"- 原始值降采样（32点，从左到右）：{wf['raw_downsample']}",
                f"- 异常分数降采样（32点，从左到右）：{wf['score_downsample']}",
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

        lines += ["", "【三层级联检测结果】"]
        if ls:
            lines += [
                f"- 最终异常分数：{ls['final_score']:.4f}"
                + (f"（L1={ls['l1_decision']}/{ls['l1_score']:.3f}）" if ls.get("l1_decision") else ""),
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
            ]

        lines += [
            "",
            f"【历史告警】过去共 {s['n_history_alerts']} 条，最近状态：{s['history_statuses'] or '无'}",
            "",
            "请判断本条告警是真实异常还是误报，并给出依据。",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM API call
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> tuple[str, str | None]:
        """Call the chat-completions endpoint. Returns (text, error)."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # temperature=0.3 (raised from 0.1) — the diagnosis prompt must
            # counter a strong prior toward "real" (selective sampling feeds
            # only alert evidence, no normal baseline).  A slightly higher
            # temperature lets the model question the premise rather than
            # defaulting to the alert-is-real bias baked into low-temp
            # decoding.  Empirically 0.1 produced 84% "real" verdicts vs
            # 30% false-alarm rate in human annotation.
            "temperature": 0.3,
            "max_tokens": 768,
            # DeepSeek models default to thinking-enabled (chain-of-thought
            # before the final answer), which doubles latency and makes
            # temperature/top_p ineffective.  Disable it — the diagnosis
            # task is straightforward and does not need reasoning traces.
            "thinking": {"type": "disabled"},
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return "", f"LLM API returned {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text.strip(), None
        except httpx.TimeoutException:
            return "", "LLM API timeout (30s)"
        except Exception as e:
            return "", f"LLM API error: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, channel: str) -> tuple[str | None, str | None]:
        """Look up display name + device-tree path for a channel."""
        try:
            tree = self.config_service.config if self.config_service else None
            if not tree:
                return None, None
            # Walk the tree to find the sensor whose sourceId maps to channel.
            from ..services.tree_utils import get_flat_sensors

            sensors = get_flat_sensors(tree)
            path_parts = []
            display = None
            # Build a channel → folder-path map by walking the tree.
            def _walk(node, prefix):
                nonlocal display
                if node.get("type") == "sensor":
                    sid = node.get("sourceId", "")
                    if sid == channel or node.get("channel") == channel:
                        display = node.get("name")
                        path_parts.extend(prefix + [node.get("name", channel)])
                else:
                    children = node.get("children", [])
                    for c in children:
                        _walk(c, prefix + [node.get("name", "")])

            root = tree.get("device_tree", tree)
            children = root.get("children", []) if isinstance(root, dict) else []
            for c in children:
                _walk(c, [])
            if display is None:
                # Fallback: linear search in flat sensors.
                for sn in sensors:
                    if sn.get("sourceId") == channel:
                        display = sn.get("name")
                        break
            return display, " / ".join(p for p in path_parts if p) or None
        except Exception:
            logger.debug("device resolve failed for %s", channel, exc_info=True)
            return None, None

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
        }

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
        return {
            "mean": mu,
            "std": sigma,
            "min": float(raws.min()),
            "max": float(raws.max()),
            "n": len(normal),
        }

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
