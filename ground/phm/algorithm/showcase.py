"""Display metadata for the algorithm/model library admin page.

This is a *display-only* layer on top of :data:`MODEL_REGISTRY` and
:data:`FILTER_REGISTRY`.  It carries no runtime logic — just human-readable
Chinese names, one-line descriptions, and parameter specs that the admin
``library`` page renders into cards.  Keeping it separate from the algorithm
modules avoids polluting the hot path with i18n strings and keeps the
display layer easy to edit when a module's user-facing copy changes.

Each :class:`ShowcaseEntry` ``key`` must match either a key in
:data:`MODEL_REGISTRY` (when ``is_model=True``) or a name in
:data:`FILTER_REGISTRY` (when ``is_model=False``).  Use
:func:`validate_showcase_consistency` at startup to detect drift.

Layer mapping to the 5 sub-menu categories:

    L1        → "L1 预处理算法库"  (5 cards)
    L2        → "L2 检测模型库"    (1 card, tspulse)
    L3        → "L3 后处理算法库"  (8 cards, incl. 3 L3.5 modules)
    forecast  → "预测模型库"       (1 card, ttm_r3)
    special   → "特殊算法或模型库"  (1 card, rul)

L3.5 modules (knee / EMA / persistence) carry ``is_l35=True`` so the
template can tag them "后处理增强" — they still live under the L3 sub-menu
to avoid adding a 6th category.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    "ShowcaseEntry",
    "SHOWCASE_REGISTRY",
    "LAYER_TO_CATEGORY",
    "validate_showcase_consistency",
]


@dataclass(frozen=True)
class ShowcaseEntry:
    """Display metadata for one algorithm/model card.

    Attributes:
        key: matches a MODEL_REGISTRY key (when ``is_model``) or a
            FILTER_REGISTRY name (otherwise).
        layer: ``"L1"`` / ``"L2"`` / ``"L3"`` / ``"forecast"`` / ``"special"``.
        display_name: Chinese name shown as the card title.
        icon: Font Awesome icon class (e.g. ``"fas fa-minus"``).
        description: one-line Chinese description of the module's effect.
        params: list of ``{name, type, default, description}`` dicts — the
            user-facing parameter signature.  Reflects the rule/model
            constructor defaults; sync when the constructor changes.
        is_model: ``True`` for model cards (sourced from MODEL_REGISTRY),
            ``False`` for algorithm cards (sourced from FILTER_REGISTRY).
        is_l35: ``True`` for the three L3.5 post-processing enhancement
            modules.  They render under the L3 sub-menu with a small
            "后处理增强" tag.
    """

    key: str
    layer: str
    display_name: str
    icon: str
    description: str
    params: list[dict] = field(default_factory=list)
    is_model: bool = False
    is_l35: bool = False


# Category-key → Chinese tab name.  Order = display order in the sub-menu.
LAYER_TO_CATEGORY = {
    "L1": "l1",
    "L2": "l2",
    "L3": "l3",
    "forecast": "forecast",
    "special": "special",
}


SHOWCASE_REGISTRY: list[ShowcaseEntry] = [
    # ── L1 preprocessing algorithms (5) ────────────────────────────────
    ShowcaseEntry(
        key="l1_constant",
        layer="L1",
        display_name="常数通道检测",
        icon="fas fa-minus",
        description="识别指令通道的常数突变，打分 1.0 并触发 L1 短路（跳过 L2）。",
        params=[
            {"name": "constant_std", "type": "float", "default": 1e-3,
             "description": "常数判定阈值（标准差低于此值视为常数）"},
            {"name": "min_finite", "type": "int", "default": 2,
             "description": "最少有效样本数"},
            {"name": "enable_constant", "type": "bool", "default": True,
             "description": "是否启用常数检测"},
        ],
        is_model=False,
    ),
    ShowcaseEntry(
        key="l1_sigma",
        layer="L1",
        display_name="σ 标准差检测",
        icon="fas fa-wave-square",
        description="基于 k·σ 的统计阈值检测，最常用的 L1 规则。",
        params=[
            {"name": "sigma_k", "type": "float", "default": 3.0,
             "description": "k 倍标准差阈值"},
            {"name": "min_sigma", "type": "float", "default": 0.0,
             "description": "σ 下限（防止恒值通道除零）"},
        ],
        is_model=False,
    ),
    ShowcaseEntry(
        key="l1_iqr",
        layer="L1",
        display_name="IQR 四分位检测",
        icon="fas fa-chart-bar",
        description="基于四分位距 (IQR) 的鲁棒统计检测，对离群值不敏感。",
        params=[
            {"name": "iqr_factor", "type": "float", "default": 1.5,
             "description": "IQR 倍数阈值"},
            {"name": "min_finite", "type": "int", "default": 20,
             "description": "最少有效样本数"},
        ],
        is_model=False,
    ),
    ShowcaseEntry(
        key="l1_rate",
        layer="L1",
        display_name="变化率检测",
        icon="fas fa-bolt",
        description="相邻样本变化率超阈视为异常，针对突变型故障。",
        params=[
            {"name": "rate_quantile", "type": "float", "default": 99.0,
             "description": "变化率分位数阈值（%）"},
            {"name": "rate_multiplier", "type": "float", "default": 5.0,
             "description": "分位数倍数"},
            {"name": "max_rate", "type": "float", "default": None,
             "description": "绝对变化率上限（可选）"},
        ],
        is_model=False,
    ),
    ShowcaseEntry(
        key="l1_setpoint",
        layer="L1",
        display_name="期望值检测",
        icon="fas fa-bullseye",
        description="基于科学家物理期望的检测（command/range/enumerate 三模式），@算法 DSL 核心依赖。",
        params=[
            {"name": "mode", "type": "str", "default": "range",
             "description": "command / range / enumerate"},
            {"name": "expected", "type": "float", "default": None,
             "description": "期望值（range 模式）"},
            {"name": "tolerance", "type": "float", "default": None,
             "description": "容差（range 模式）"},
            {"name": "command_value", "type": "float", "default": None,
             "description": "指令值（command 模式）"},
            {"name": "anomaly_values", "type": "list", "default": None,
             "description": "异常值列表（command 模式）"},
            {"name": "legal_values", "type": "list", "default": None,
             "description": "合法枚举值（enumerate 模式）"},
        ],
        is_model=False,
    ),

    # ── L2 detection models (1) ────────────────────────────────────────
    ShowcaseEntry(
        key="tspulse",
        layer="L2",
        display_name="TSPulse 异常检测",
        icon="fas fa-search",
        description="零样本时序异常检测模型（TSB-UAD 基准），L2 层默认检测器。",
        params=[
            {"name": "context_length", "type": "int", "default": 512,
             "description": "输入窗口长度（点）"},
            {"name": "hub_id", "type": "str",
             "default": "ibm-granite/granite-timeseries-tspulse-r1",
             "description": "HuggingFace 模型 id"},
        ],
        is_model=True,
    ),

    # ── L3 post-processing algorithms (5 physical constraints) ────────
    ShowcaseEntry(
        key="l3_nan_sanitise",
        layer="L3",
        display_name="NaN 净化",
        icon="fas fa-broom",
        description="L3 首道：将 NaN / inf 样本置为安全值，避免污染后续规则。",
        params=[],
        is_model=False,
    ),
    ShowcaseEntry(
        key="l3_constant",
        layer="L3",
        display_name="L3 常数抑制",
        icon="fas fa-grip-lines",
        description="常数通道短路：恒值段直接置零分，避免误报。",
        params=[
            {"name": "constant_std", "type": "float", "default": 1e-3,
             "description": "常数判定阈值"},
            {"name": "min_finite", "type": "int", "default": 2,
             "description": "最少有效样本数"},
        ],
        is_model=False,
    ),
    ShowcaseEntry(
        key="l3_range",
        layer="L3",
        display_name="量程约束",
        icon="fas fa-ruler-horizontal",
        description="物理量程越界检测：超 [valid_min, valid_max] 视为异常。",
        params=[
            {"name": "valid_min", "type": "float", "default": None,
             "description": "量程下限"},
            {"name": "valid_max", "type": "float", "default": None,
             "description": "量程上限"},
            {"name": "range_boost", "type": "float", "default": 0.95,
             "description": "越界分数增益"},
        ],
        is_model=False,
    ),
    ShowcaseEntry(
        key="l3_rate",
        layer="L3",
        display_name="L3 变化率约束",
        icon="fas fa-angle-double-up",
        description="变化率超物理上限视为异常，抑制尖刺型误报。",
        params=[
            {"name": "max_rate", "type": "float", "default": None,
             "description": "最大允许变化率"},
            {"name": "rate_boost", "type": "float", "default": 0.85,
             "description": "变化率异常分数增益"},
        ],
        is_model=False,
    ),
    ShowcaseEntry(
        key="l3_variance",
        layer="L3",
        display_name="方差阻尼",
        icon="fas fa-compress-arrows-alt",
        description="高方差段（噪声）分数阻尼，避免噪声通道频繁告警。",
        params=[
            {"name": "baseline_var", "type": "float", "default": None,
             "description": "基线方差"},
            {"name": "var_dampen_ratio", "type": "float", "default": 10.0,
             "description": "方差放大倍数触发阻尼"},
            {"name": "var_dampen_factor", "type": "float", "default": 0.3,
             "description": "阻尼系数"},
        ],
        is_model=False,
    ),

    # ── L3.5 post-processing enhancements (3, shown under L3 sub-menu) ──
    ShowcaseEntry(
        key="l3_knee_threshold",
        layer="L3",
        display_name="Knee 自适应阈值",
        icon="fas fa-project-diagram",
        description="基于拐点法的无泄漏自适应阈值，在线累积满 512 点自动推导。",
        params=[
            {"name": "eps", "type": "float", "default": 1e-6,
             "description": "零分阈值（小于此值视为 0）"},
            {"name": "min_fit_samples", "type": "int", "default": 512,
             "description": "推导阈值所需最小累积样本数"},
            {"name": "threshold_override", "type": "float", "default": None,
             "description": "固定阈值覆盖（可选，跳过在线推导）"},
        ],
        is_model=False,
        is_l35=True,
    ),
    ShowcaseEntry(
        key="l3_ema_smoothing",
        layer="L3",
        display_name="EMA 指数平滑",
        icon="fas fa-water",
        description="因果 EMA 平滑（α=0.2），消除块边界毛刺，提升 VUS-PR。",
        params=[
            {"name": "alpha", "type": "float", "default": 0.2,
             "description": "平滑系数 (0,1]，越小越平滑"},
        ],
        is_model=False,
        is_l35=True,
    ),
    ShowcaseEntry(
        key="l3_persistence",
        layer="L3",
        display_name="W/K 持久性滤波",
        icon="fas fa-shield-alt",
        description="滑窗内至少 K 个样本越阈才告警，去除瞬时毛刺。",
        params=[
            {"name": "W", "type": "int", "default": 5,
             "description": "持久性窗口（样本数）"},
            {"name": "K", "type": "int", "default": 3,
             "description": "窗口内最少越阈样本数"},
            {"name": "threshold", "type": "float", "default": 0.5,
             "description": "二值化分数阈值"},
        ],
        is_model=False,
        is_l35=True,
    ),

    # ── Forecast models (1) ────────────────────────────────────────────
    ShowcaseEntry(
        key="ttm_r3",
        layer="forecast",
        display_name="TTM-R3 趋势预测",
        icon="fas fa-chart-line",
        description="零样本时序预测（512→96），地基趋势预测默认模型。",
        params=[
            {"name": "context_length", "type": "int", "default": 512,
             "description": "输入窗口长度（点）"},
            {"name": "prediction_length", "type": "int", "default": 96,
             "description": "预测步长（点）"},
            {"name": "hub_id", "type": "str", "default": "ibm-research/ttm-r3",
             "description": "HuggingFace 模型 id"},
        ],
        is_model=True,
    ),

    # ── Special algorithms / models (1) ────────────────────────────────
    ShowcaseEntry(
        key="rul",
        layer="special",
        display_name="RUL 退化预测",
        icon="fas fa-battery-half",
        description="LSTM+Attention 监督式剩余寿命预测（C-MAPSS FD001 RMSE=14.88），本地权重。",
        params=[
            {"name": "context_length", "type": "int", "default": 30,
             "description": "输入窗口长度（点）"},
            {"name": "weights_dir", "type": "str", "default": "models/rul/",
             "description": "本地权重目录"},
        ],
        is_model=True,
    ),
]


def validate_showcase_consistency() -> list[str]:
    """Check that every ShowcaseEntry key resolves in the matching registry.

    Returns a list of human-readable warning strings (empty when consistent).
    Logs each warning at WARNING level so a misconfigured deploy surfaces in
    the server log without crashing startup (display metadata is non-critical
    — a missing entry just means the admin card has no metadata).

    Call this once at Django startup (e.g. in ``AppConfig.ready``) or from a
    management command.  The check is cheap (dict membership, no imports of
    torch / model loaders).
    """
    warnings: list[str] = []

    # Import lazily so importing this module never pulls in the rule/model
    # registries (keeps the display layer importable from contexts that do
    # not want the algorithm side effects).
    from ._registry import MODEL_REGISTRY
    from .rules import FILTER_REGISTRY

    model_keys = set(MODEL_REGISTRY.keys())
    filter_keys = set(FILTER_REGISTRY.keys())

    for entry in SHOWCASE_REGISTRY:
        if entry.is_model:
            if entry.key not in model_keys:
                msg = (
                    f"showcase entry {entry.key!r} (is_model=True) not found "
                    f"in MODEL_REGISTRY — algorithm landing checklist out of sync"
                )
                warnings.append(msg)
                logger.warning(msg)
        else:
            if entry.key not in filter_keys:
                msg = (
                    f"showcase entry {entry.key!r} (is_model=False) not found "
                    f"in FILTER_REGISTRY — algorithm landing checklist out of sync"
                )
                warnings.append(msg)
                logger.warning(msg)

    return warnings
