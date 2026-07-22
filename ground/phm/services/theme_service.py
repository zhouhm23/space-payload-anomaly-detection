"""Front-end theme / display configuration service.

Loads ``data/ui_theme.json`` once and exposes it to the Django template
layer via a context processor (synchronous injection into ``window.THEME``).
This is the front-end counterpart to :class:`SystemConfigService` — same
load-once-with-fallback pattern, different consumer (browser vs. services).

Why a service (not just a static JSON read)?
  * Centralises the default fallbacks so the front-end never breaks on a
    malformed or missing theme file.
  * ``_strip_docs`` keeps ``_doc`` annotation keys out of the payload sent
    to the browser (they are author notes, not runtime data).
  * Future hot-reload / per-user themes can hook in here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ThemeService", "get_theme", "reset_theme"]


# Default location: src/ground/data/ui_theme.json
_DEFAULT_THEME_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "ui_theme.json",
)

# Fallback defaults — the exact values monitor.js used to hard-code. Used
# when the JSON is missing or malformed so the page always renders.
_DEFAULTS: dict[str, Any] = {
    "colors": {
        "bgPrimary": "#0b0f1a", "bgSecondary": "#131825", "bgCard": "#1a1f2e",
        "border": "#2a3348", "textPri": "#e0e6f0", "textSec": "#8e9bb5",
        "blue": "#2d8cf0", "green": "#19be6b", "yellow": "#f5a623",
        "red": "#ed3f14", "cyan": "#00c9db",
    },
    "thresholds": {
        "anomalyScoreRed": 0.5, "anomalyScoreYellow": 0.25,
        "healthRed": 60, "healthYellow": 80,
        "rulGreen": 0.6, "rulYellow": 0.25,
    },
    "poll": {
        "chart": 2000, "health": 3000, "sensors": 3000,
        "alerts": 3000, "warnings": 3000, "rul": 5000,
        "dbStats": 5000, "diagnosis": 2000,
    },
    "chart": {
        "cacheCount": 2048, "viewCount": 512, "prefetchThreshold": 256,
        "topRatio": 0.7,
        "padding": {"top": 20, "right": 50, "bottom": 30, "left": 60},
        "gapWidthPx": 40,
    },
    "display": {
        "systemTitle": "空间站有效载荷预测性维护支持系统",
        "clockTimezone": "Asia/Shanghai",
        "datetimeFormat": "YYYY-MM-DD HH:MM:SS UTC",
    },
    "layout": {
        "headerHeight": 60, "leftPanelWidth": 240,
        "rightPanelWidth": 340, "bottomPanelFlex": 1.4,
    },
    "network": {"linkFailThreshold": 3},
}

# Keys that are documentation-only and must not reach the browser.
_DOC_KEYS = {"_doc"}

# 中文展示名映射（后台「系统设置 · 前台主题」用）。
# 与 ui_theme.json 的 _doc 互补——_doc 是悬浮说明，这里是列表 label。
_DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "colors": {
        "_doc": "调色板",
        "bgPrimary": "背景主色", "bgSecondary": "背景次色", "bgCard": "卡片背景",
        "border": "边框色", "textPri": "主文字", "textSec": "次文字",
        "blue": "蓝", "green": "绿", "yellow": "黄", "red": "红", "cyan": "青",
    },
    "thresholds": {
        "_doc": "分数/健康度色界",
        "anomalyScoreRed": "异常分数红界", "anomalyScoreYellow": "异常分数黄界",
        "healthRed": "健康度红界", "healthYellow": "健康度黄界",
        "rulGreen": "RUL 绿界", "rulYellow": "RUL 黄界",
    },
    "poll": {
        "_doc": "轮询间隔（毫秒）",
        "chart": "图表", "health": "健康度", "sensors": "传感器列表",
        "alerts": "告警", "warnings": "预警", "rul": "RUL",
        "dbStats": "数据库统计", "diagnosis": "诊断",
    },
    "chart": {
        "_doc": "图表尺寸",
        "cacheCount": "缓存点数", "viewCount": "可视点数",
        "prefetchThreshold": "预取阈值", "topRatio": "顶部比例",
        "gapWidthPx": "间隔宽度（px）",
    },
    "display": {
        "_doc": "界面文案/时区",
        "systemTitle": "系统标题",
        "clockTimezone": "时钟时区",
        "datetimeFormat": "日期时间格式",
    },
    "layout": {
        "_doc": "面板几何（px/flex）",
        "headerHeight": "头部高度",
        "carouselBarHeight": "轮播条高度",
        "leftPanelWidth": "左面板宽度",
        "rightPanelWidth": "右面板宽度",
        "bottomPanelFlex": "底面板 flex",
    },
    "chart412": {
        "_doc": "中央图表 4:1:2 比例",
        "topRatio": "顶部比例", "midRatio": "中部比例", "bottomRatio": "底部比例",
        "gapWidthPx": "间隔宽度（px）", "predOnlyMinPoints": "预测最小点数",
        "defaultYMin": "默认 Y 下限", "defaultYMax": "默认 Y 上限",
    },
    "carousel": {
        "_doc": "通道轮播",
        "intervalMs": "切换间隔（毫秒）",
        "manualInteractionPauseMs": "手动操作暂停时长（毫秒）",
    },
    "network": {
        "_doc": "前台链路状态",
        "linkFailThreshold": "链路中断阈值",
    },
}

# 嵌套 dict 类型（如 chart.padding / chart412.padding）暂不支持网页编辑。
# 这些 key 父级是 dict 而非标量，UI 应整体灰显。
_NESTED_KEYS = frozenset({
    "chart.padding", "chart412.padding",
})


def _strip_docs(obj: Any) -> Any:
    """Recursively remove ``_doc`` keys from a nested dict/list structure."""
    if isinstance(obj, dict):
        return {k: _strip_docs(v) for k, v in obj.items() if k not in _DOC_KEYS}
    if isinstance(obj, list):
        return [_strip_docs(item) for item in obj]
    return obj


class ThemeService:
    """Load-once reader for ``ui_theme.json`` with built-in fallbacks."""

    def __init__(self, theme_path: str | None = None) -> None:
        self.theme_path = theme_path or _DEFAULT_THEME_PATH
        self._theme: dict[str, Any] = {}
        self._lock = Lock()
        self.load()

    def load(self) -> None:
        """(Re)load the JSON. Missing/malformed → fall back to defaults."""
        with self._lock:
            # Deep-copy defaults so mutation can't leak back.
            self._theme = json.loads(json.dumps(_DEFAULTS))
        if not os.path.exists(self.theme_path):
            logger.debug(
                "ui_theme.json not found at %s — using built-in defaults",
                self.theme_path,
            )
            return
        try:
            with open(self.theme_path, encoding="utf-8") as f:
                raw = json.load(f)
            with self._lock:
                # Merge: JSON overrides defaults section by section.
                for section, values in raw.items():
                    if section in _DOC_KEYS or not isinstance(values, dict):
                        continue
                    base = self._theme.setdefault(section, {})
                    base.update({
                        k: v for k, v in values.items() if k not in _DOC_KEYS
                    })
            logger.info("loaded ui_theme from %s", self.theme_path)
        except Exception:
            logger.warning(
                "failed to load ui_theme %s — using defaults",
                self.theme_path, exc_info=True,
            )

    def reload(self) -> None:
        self.load()

    def as_dict(self) -> dict[str, Any]:
        """Return the theme with ``_doc`` keys stripped (for browser injection)."""
        with self._lock:
            return _strip_docs(self._theme)

    # ── Back-office write support (settings page) ──────────────────────

    def raw_with_docs(self) -> dict[str, Any]:
        """读取磁盘 JSON **原文**（含 ``_doc``），供后台 UI 渲染悬浮描述。

        与 ``as_dict()`` 的区别：as_dict 已 strip _doc；本方法直接 open →
        json.load，保证 UI 看到的是文件最新状态。文件缺失时回退到
        _DEFAULTS 深拷贝（带 _DISPLAY_NAMES 的 _doc，保证 UI 仍能渲染）。
        """
        try:
            with open(self.theme_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return raw
        except FileNotFoundError:
            logger.debug("ui_theme.json not found at %s — using defaults", self.theme_path)
        except Exception:
            logger.warning("failed to read raw ui_theme %s — using defaults",
                           self.theme_path, exc_info=True)
        # 回退：_DEFAULTS + _DISPLAY_NAMES _doc
        out: dict[str, Any] = json.loads(json.dumps(_DEFAULTS))
        for section, names in _DISPLAY_NAMES.items():
            doc = names.get("_doc")
            if doc and section in out and isinstance(out[section], dict):
                out[section]["_doc"] = doc
        return out

    def is_readonly(self, section: str, key: str) -> bool:
        """该 key 是否只读（嵌套 dict 类型的子键、或不在 _DEFAULTS 的 key）。"""
        if f"{section}.{key}" in _NESTED_KEYS:
            return True
        defaults_section = _DEFAULTS.get(section)
        if not isinstance(defaults_section, dict):
            return True
        # 嵌套 dict 子键（如 chart.padding）：父级非标量 → 整体只读
        existing = defaults_section.get(key)
        if isinstance(existing, dict):
            return True
        return key not in defaults_section

    @staticmethod
    def display_names() -> dict[str, dict[str, str]]:
        """返回中文展示名映射（后台 UI 用）。"""
        return _DISPLAY_NAMES

    def save(self, section: str, key: str, value: Any) -> dict[str, Any]:
        """更新单个 key 的值，写回 JSON 并热生效（重新 load）。

        嵌套 dict 类型的子键（如 ``chart.padding.top``）不在本方法范围内——
        UI 直接整体灰显，不允许网页编辑。

        Returns:
            ``{"status": "ok", "old": ..., "new": ...}`` 成功；
            ``{"status": "error", "message": ...}`` 失败。
        """
        # 1) 合法性
        if section in _DOC_KEYS or not isinstance(section, str) or not section:
            return {"status": "error", "message": f"非法 section：{section!r}"}
        if key in _DOC_KEYS or not isinstance(key, str) or not key:
            return {"status": "error", "message": f"非法 key：{key!r}"}
        if f"{section}.{key}" in _NESTED_KEYS:
            return {"status": "error", "message": f"{section}.{key} 为嵌套对象，不支持网页编辑"}
        defaults_section = _DEFAULTS.get(section)
        if not isinstance(defaults_section, dict) or key not in defaults_section:
            return {"status": "error",
                    "message": f"未知配置项：{section}.{key}（不在 _DEFAULTS 中）"}
        expected = defaults_section[key]
        if isinstance(expected, dict):
            return {"status": "error",
                    "message": f"{section}.{key} 为嵌套对象，不支持网页编辑"}

        # 2) 类型校验（同 SystemConfigService 范式）
        if isinstance(expected, bool):
            if not isinstance(value, bool):
                return {"status": "error",
                        "message": f"{section}.{key} 期望 bool，实际 {type(value).__name__}"}
        elif isinstance(expected, int) and not isinstance(expected, bool):
            if not isinstance(value, int) or isinstance(value, bool):
                return {"status": "error",
                        "message": f"{section}.{key} 期望 int，实际 {type(value).__name__}"}
        elif isinstance(expected, float):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return {"status": "error",
                        "message": f"{section}.{key} 期望 float，实际 {type(value).__name__}"}
        elif isinstance(expected, str):
            if not isinstance(value, str):
                return {"status": "error",
                        "message": f"{section}.{key} 期望 str，实际 {type(value).__name__}"}
        else:
            return {"status": "error",
                    "message": f"{section}.{key} 类型 {type(expected).__name__} 暂不支持网页编辑"}

        # 3) 原子写回 + 热生效
        try:
            raw = self.raw_with_docs()
            sec = raw.setdefault(section, {})
            if not isinstance(sec, dict):
                return {"status": "error",
                        "message": f"section {section!r} 不是对象"}
            old_value = sec.get(key)
            sec[key] = value
            self._atomic_write(raw)
        except Exception as e:
            logger.warning("theme save failed: %s", e, exc_info=True)
            return {"status": "error", "message": f"写盘失败：{e}"}

        try:
            self.load()
        except Exception as e:
            logger.warning("theme reload after save failed: %s", e, exc_info=True)

        return {"status": "ok", "section": section, "key": key,
                "old": old_value, "new": value}

    def _atomic_write(self, data: dict[str, Any]) -> None:
        """原子写回 JSON：临时文件 → os.replace 覆盖（同 SystemConfigService）。"""
        d = os.path.dirname(self.theme_path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".ui_theme_", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp_path, self.theme_path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise


# ── Process-wide singleton ─────────────────────────────────────────────

_singleton: ThemeService | None = None
_singleton_lock = Lock()


def get_theme() -> ThemeService:
    """Return the process-wide :class:`ThemeService` singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ThemeService()
    return _singleton


def reset_theme() -> None:
    """Drop the singleton (test helper)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
