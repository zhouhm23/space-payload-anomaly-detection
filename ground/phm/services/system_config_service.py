"""System-wide runtime configuration service.

Loads ``data/system_config.json`` once and exposes typed accessors for every
value that ``phm/config.py`` previously hard-coded.  This is the *single
source of truth* at runtime — ``config.py`` now reads from here lazily via
module-level ``__getattr__`` (PEP 562), so existing ``from phm.config import
ANOMALY_THRESHOLD`` imports keep working without any caller change.

Design (mirrors ``CalibrationConfig``):
  * Constructor takes an optional path; falls back to the default location.
  * ``load()`` reads the JSON; a missing or malformed file logs a warning
    and falls back to ``_DEFAULTS`` (the code never crashes on bad config).
  * Each section is exposed as a property returning a dict; individual
    values via ``get(section, key)`` for ad-hoc access.
  * Agent-friendly: ``snapshot()`` returns the whole config as a dict for
    ``manage.py config`` and ``GET /api/config/system``.

The defaults embedded here are the exact values that were hard-coded in
``config.py`` before this refactor — so behaviour is identical when the JSON
is absent.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SystemConfigService", "get_system_config", "reset_system_config",
           "DEFAULT_CONFIG_PATH"]


# Default location: src/ground/data/system_config.json
# Path from here (src/ground/phm/services/): up 3 → src/ground/, then data/.
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "system_config.json",
)
# Backwards-compat alias (older code referenced the underscore-prefixed name).
_DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH

# Fallback defaults — the exact values config.py used to hard-code.  Used
# when the JSON is missing or a key is absent so the system always runs.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "network": {
        "space_host": "127.0.0.1",
        "space_port": 9876,
        "ground_port": 8501,
        "link_fail_threshold": 3,
    },
    "storage": {
        "db_path": "data/phm.db",
        "ring_buffer_max": 20000,
        "sqlite_batch_size": 200,
        "sqlite_flush_interval_sec": 2.0,
        "sqlite_enabled": True,
    },
    "thresholds": {
        "anomaly": 0.5,
        "l1_constant_std": 1e-3,
        "l1_sigma_k": 3.0,
        "l1_iqr_factor": 1.5,
        "l3_constant_std": 1e-3,
        "l3_range_boost": 0.95,
        "l3_rate_boost": 0.85,
    },
    "forecast": {
        "context_length": 512,
        "prediction_length": 96,
    },
    "warning": {
        "min_predict_scores": 1,
    },
    "rul": {
        "enabled": True,
        "window_cycles": 30,
        "history_len": 20,
        "poll_interval_sec": 5.0,
    },
    "llm": {
        "timeout_sec": 30.0,
    },
}

# Keys that are documentation-only and should never be surfaced as config.
_DOC_KEYS = {"_doc"}

# Read-only keys (managed by .env or other deployment mechanisms; the UI must
# grey these out instead of allowing edits). Listed by "section.key".
_READONLY_KEYS = frozenset({
    "llm.timeout_sec",  # API key / base_url / model_name come from .env
})

# 中文展示名映射（后台「系统设置」页用）。结构：
#   {section: {"_doc": "<section 中文标题>", "<key>": "<中文展示名>"}}
# 与 data/system_config.json 的 _doc 字段互补——_doc 是悬浮说明，这里是
# 列表里的中文 label。两者一起在 UI 上呈现：中文名 + (悬浮) 描述。
_DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "network": {
        "_doc": "网络配置",
        "space_host": "空间段地址",
        "space_port": "空间段端口",
        "ground_port": "地面段端口",
        "link_fail_threshold": "链路中断阈值（连续失败次数）",
    },
    "storage": {
        "_doc": "存储配置",
        "db_path": "SQLite 数据库路径",
        "ring_buffer_max": "环形缓冲区容量",
        "sqlite_batch_size": "SQLite 批量写入条数",
        "sqlite_flush_interval_sec": "SQLite 刷新间隔（秒）",
        "sqlite_enabled": "SQLite 持久化开关",
    },
    "thresholds": {
        "_doc": "异常检测阈值",
        "anomaly": "异常分数阈值",
        "l1_constant_std": "L1 常数标准差阈值",
        "l1_sigma_k": "L1 σ 倍数",
        "l1_iqr_factor": "L1 IQR 因子",
        "l3_constant_std": "L3 常数标准差阈值",
        "l3_range_boost": "L3 幅值增强系数",
        "l3_rate_boost": "L3 变化率增强系数",
    },
    "forecast": {
        "_doc": "TTM-R3 预测参数",
        "context_length": "上下文长度",
        "prediction_length": "预测长度",
    },
    "warning": {
        "_doc": "预测预警参数",
        "min_predict_scores": "触发预警最少预测点数",
    },
    "rul": {
        "_doc": "退化预测（RUL）参数",
        "enabled": "启用 RUL",
        "window_cycles": "窗口周期数",
        "history_len": "历史长度",
        "poll_interval_sec": "轮询间隔（秒）",
    },
    "llm": {
        "_doc": "LLM 诊断参数",
        "timeout_sec": "LLM 调用超时（秒）",
    },
}


class SystemConfigService:
    """Typed, reloadable reader for ``system_config.json``.

    Thread-safe (a background reload could be added later; for now load()
    is called once at construction).
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._cfg: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self.load()

    def load(self) -> None:
        """(Re)load the JSON. Missing file → use ``_DEFAULTS`` entirely.
        Malformed file → log warning and keep defaults."""
        with self._lock:
            self._cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
        if not os.path.exists(self.config_path):
            logger.debug(
                "system_config.json not found at %s — using built-in defaults",
                self.config_path,
            )
            return
        try:
            with open(self.config_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Merge: JSON overrides defaults section by section, key by key.
            # Unknown sections/keys are kept (forward-compat) but _doc keys
            # are stripped from the typed view.
            with self._lock:
                for section, values in raw.items():
                    if section in _DOC_KEYS or not isinstance(values, dict):
                        continue
                    base = self._cfg.setdefault(section, {})
                    for k, v in values.items():
                        if k not in _DOC_KEYS:
                            base[k] = v
            logger.info(
                "loaded system config from %s (%d sections)",
                self.config_path, len(self._cfg),
            )
        except Exception:
            logger.warning(
                "failed to load system config %s — using defaults",
                self.config_path, exc_info=True,
            )

    def reload(self) -> None:
        """Alias for :meth:`load` (hot-reload use case)."""
        self.load()

    # ── Typed accessors (one per config section) ───────────────────────

    @property
    def network(self) -> dict[str, Any]:
        return self._cfg["network"]

    @property
    def storage(self) -> dict[str, Any]:
        return self._cfg["storage"]

    @property
    def thresholds(self) -> dict[str, Any]:
        return self._cfg["thresholds"]

    @property
    def forecast(self) -> dict[str, Any]:
        return self._cfg["forecast"]

    @property
    def warning(self) -> dict[str, Any]:
        return self._cfg["warning"]

    @property
    def rul(self) -> dict[str, Any]:
        return self._cfg["rul"]

    @property
    def llm(self) -> dict[str, Any]:
        return self._cfg["llm"]

    # ── Ad-hoc access ──────────────────────────────────────────────────

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Return ``config[section][key]``, or ``default`` if absent."""
        return self._cfg.get(section, {}).get(key, default)

    def is_readonly(self, section: str, key: str) -> bool:
        """该 key 是否只读（环境变量管理等场景，UI 应灰显）。"""
        return f"{section}.{key}" in _READONLY_KEYS

    @staticmethod
    def display_names() -> dict[str, dict[str, str]]:
        """返回中文展示名映射（后台 UI 用）。"""
        return _DISPLAY_NAMES

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the full config (for CLI / API exposure).

        Strips ``_doc`` keys — those are author annotations, not runtime data.
        """
        with self._lock:
            return {
                section: {k: v for k, v in values.items() if k not in _DOC_KEYS}
                for section, values in self._cfg.items()
            }

    # ── Back-office write support (settings page) ──────────────────────

    def raw_with_docs(self) -> dict[str, dict[str, Any]]:
        """读取磁盘 JSON **原文**（含 ``_doc`` 字段），供后台 UI 渲染悬浮描述。

        与 ``snapshot()`` 的区别：snapshot 运行时值（已 strip _doc）；本方法
        直接 ``open → json.load``，每次调用都从磁盘重新读，保证 UI 看到的
        是文件最新状态（save 写入后立即可见）。文件缺失时回退到 _DEFAULTS
        深拷贝（不抛异常）。

        返回结构：与原始 JSON 同构，每个 section 是 ``{_doc: str, key: value, ...}``。
        """
        try:
            with open(self.config_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return raw
        except FileNotFoundError:
            logger.debug("system_config.json not found at %s — using defaults", self.config_path)
        except Exception:
            logger.warning("failed to read raw system config %s — using defaults",
                           self.config_path, exc_info=True)
        # 回退：合并 _DEFAULTS + _DISPLAY_NAMES 的 _doc（保证 UI 仍能渲染）
        out: dict[str, dict[str, Any]] = {}
        for section, values in _DEFAULTS.items():
            entry: dict[str, Any] = {}
            doc = _DISPLAY_NAMES.get(section, {}).get("_doc")
            if doc:
                entry["_doc"] = doc
            entry.update(values)
            out[section] = entry
        return out

    def save(self, section: str, key: str, value: Any) -> dict[str, Any]:
        """更新单个 key 的值，写回 JSON 并热生效（重新 load）。

        Args:
            section: 顶层 section 名（如 ``"thresholds"``）。
            key: section 下的 key（如 ``"anomaly"``）。
            value: 新值。类型必须与 _DEFAULTS[section][key] 一致（int/float/
                bool/str），否则返回 ``type_mismatch``。

        Returns:
            ``{"status": "ok", "old": ..., "new": ...}`` 成功；
            ``{"status": "error", "message": ...}`` 失败（未知 section/key、
            类型不匹配、写盘失败、只读 key）。
        """
        # 1) 合法性校验
        if section in _DOC_KEYS or not isinstance(section, str) or not section:
            return {"status": "error", "message": f"非法 section：{section!r}"}
        if key in _DOC_KEYS or not isinstance(key, str) or not key:
            return {"status": "error", "message": f"非法 key：{key!r}"}
        if f"{section}.{key}" in _READONLY_KEYS:
            return {"status": "error", "message": f"{section}.{key} 为只读（环境变量管理）"}
        defaults_section = _DEFAULTS.get(section)
        if not isinstance(defaults_section, dict) or key not in defaults_section:
            return {"status": "error",
                    "message": f"未知配置项：{section}.{key}（不在 _DEFAULTS 中）"}

        # 2) 类型校验（按 _DEFAULTS 推断期望类型；bool 必须严格匹配，不能是 int）
        expected = defaults_section[key]
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
            # 其他类型（list/dict 等）暂不支持网页编辑，统一拒绝
            return {"status": "error",
                    "message": f"{section}.{key} 类型 {type(expected).__name__} 暂不支持网页编辑"}

        # 3) 读原文（保留 _doc），覆盖单 key，原子写回
        try:
            raw = self.raw_with_docs()
            sec = raw.setdefault(section, {})
            old_value = sec.get(key)
            sec[key] = value
            self._atomic_write(raw)
        except Exception as e:
            logger.warning("system_config save failed: %s", e, exc_info=True)
            return {"status": "error", "message": f"写盘失败：{e}"}

        # 4) 热生效：重新 load 让运行时属性（self.thresholds 等）同步更新
        try:
            self.load()
        except Exception as e:
            logger.warning("system_config reload after save failed: %s", e, exc_info=True)

        return {"status": "ok", "section": section, "key": key,
                "old": old_value, "new": value}

    def _atomic_write(self, data: dict[str, Any]) -> None:
        """原子写回 JSON：写临时文件 → os.replace 覆盖。

        os.replace 在同一文件系统上是原子的（POSIX rename / Win MoveFileEx），
        避免写一半进程崩溃导致配置文件损坏。同一目录保证同盘。
        """
        d = os.path.dirname(self.config_path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".system_config_", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            # Windows 上 os.replace 也能覆盖已存在文件（Python 3.3+）
            os.replace(tmp_path, self.config_path)
        except Exception:
            # 清理临时文件，避免垃圾堆积
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise


# ── Process-wide singleton ─────────────────────────────────────────────
# Lazily constructed on first access so importing this module is cheap.
# ``config.py``'s ``__getattr__`` calls ``get_system_config()`` on demand.

_singleton: SystemConfigService | None = None
_singleton_lock = Lock()


def get_system_config() -> SystemConfigService:
    """Return the process-wide :class:`SystemConfigService` singleton."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = SystemConfigService(_DEFAULT_CONFIG_PATH)
    return _singleton


def reset_system_config() -> None:
    """Drop the singleton (test helper — forces re-creation on next access)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
