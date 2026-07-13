"""LLM-powered anomaly diagnosis service.

Aggregates the per-channel detection context (three-layer cascade output,
recent telemetry statistics, historical alerts, device-tree position,
offline calibration) into a structured prompt, calls an OpenAI-compatible
chat-completions API, and returns a Markdown diagnosis report covering
root-cause analysis and trend assessment.

The service is **on-demand** — it only runs when a user requests a
diagnosis for a specific alert/warning, never automatically.  Diagnosis
results are not persisted (each call is fresh).

Configuration is via standard environment variables so any
OpenAI-compatible provider works::

    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://api.deepseek.com/v1   # or any compatible endpoint
    LLM_MODEL=deepseek-chat                        # default chat model
"""

from __future__ import annotations

import logging
import os
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


_SYSTEM_PROMPT = """你是航天器有效载荷健康管理（PHM）专家。根据检测数据给出异常诊断报告。

直接输出报告，不要复述输入数据，不要展示思考过程。报告格式（Markdown）：

## 根因分析
推测异常原因，分类为：传感器故障 / 数据漂移 / 真实异常 / 模型误报。一句话给结论 + 关键依据。

## 趋势评估
分数走势、是否恶化、是否需干预。两三句话。

## 置信度
高/中/低 + 一句话理由。

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

    @property
    def enabled(self) -> bool:
        """Whether the service has the credentials to call the LLM API."""
        return bool(self.base_url and self.api_key and self.model)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def diagnose(self, channel: str, alert_type: str = "measured",
                 alert_ts: float | None = None) -> dict:
        """Produce a diagnosis report for one channel.

        Args:
            channel: telemetry channel name (e.g. ``"C-1"``).
            alert_type: ``"measured"`` (space-side score) or ``"predicted"``
                (ground-side cascade).  Determines which context to emphasise.
            alert_ts: the alert/warning timestamp — used as the cache key
                so repeated clicks on the same alert return the stored
                diagnosis without re-calling the LLM.

        Returns:
            ``{"channel", "alert_type", "diagnosis", "context_summary",
              "elapsed_sec", "error", "cached"}``.  On failure
            ``diagnosis`` is empty and ``error`` carries the reason.
        """
        # Cache lookup — one diagnosis per unique (channel, type, alert_ts).
        if alert_ts is not None and self.sqlite is not None:
            cached = self.sqlite.get_diagnosis(channel, alert_type, alert_ts)
            if cached is not None and cached.get("diagnosis"):
                return {
                    "channel": channel,
                    "alert_type": alert_type,
                    "diagnosis": cached["diagnosis"],
                    "context_summary": cached.get("context_summary", {}),
                    "elapsed_sec": cached.get("elapsed_sec", 0.0),
                    "error": cached.get("error"),
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
                "cached": False,
            }

        context = self._build_context(channel, alert_type)
        if context is None:
            return {
                "channel": channel,
                "alert_type": alert_type,
                "diagnosis": "",
                "context_summary": {},
                "elapsed_sec": time.time() - t0,
                "error": f"no detection data available for channel {channel}",
                "cached": False,
            }

        user_prompt = self._build_prompt(channel, context)
        text, err = self._call_llm(user_prompt)
        elapsed = round(time.time() - t0, 2)
        result = {
            "channel": channel,
            "alert_type": alert_type,
            "diagnosis": text,
            "context_summary": context["summary"],
            "elapsed_sec": elapsed,
            "error": err,
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
            )
        return result

    # ------------------------------------------------------------------
    # Context aggregation
    # ------------------------------------------------------------------

    def _build_context(self, channel: str, alert_type: str) -> dict | None:
        """Gather all data points the diagnosis prompt needs.

        Returns ``None`` if there is no detection data at all for the
        channel (the LLM would have nothing to analyse).
        """
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

        if cascade_dict is None and not recent_rows:
            return None

        # Compact numeric summary of the recent window (avoid dumping 512
        # raw numbers into the prompt).
        window_stats = self._summarise_window(recent_rows)

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
            f"告警类型：{'实测告警（space段TSPulse）' if s['alert_type']=='measured' else '预测预警（ground段级联）'}",
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
            lines += [
                "",
                f"【未来 {s['predict_horizon']} 点预测】",
                f"- 预测段最大异常分数：{s['predict_max_score']:.4f}"
                + ("（超过阈值，预示恶化）" if s["predict_max_score"] > s["threshold"] else "（低于阈值）"),
            ]

        lines += [
            "",
            f"【历史告警】过去共 {s['n_history_alerts']} 条，最近状态：{s['history_statuses'] or '无'}",
            "",
            "请给出诊断报告。",
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
            "temperature": 0.1,
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
