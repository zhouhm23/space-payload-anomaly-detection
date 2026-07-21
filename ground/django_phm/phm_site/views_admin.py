"""后台自定义页视图（/admin/phm_site/<page>/）。

需求书 §后台 共 9 个页面：
  - 登录 / 首页 / 用户管理 / 审计日志：走 SimpleUI 默认，本文件不涉及
  - 仪表盘 / 告警与预警 / 回收站 / 设备树 / 系统设置 / 模型管理：本文件实现

设计要点：
  - 所有页面 view 用 ``@staff_member_required`` 守门（需求书"没登录显示登录页"）
  - 复用 ``views_api._container_or_503`` 的三态机思路，但返回 Django HttpResponse
  - AJAX 操作（标注/删除/保存）同文件内 JSON view，路径形如
    ``/admin/phm_site/<page>/api/<action>/``
  - 业务逻辑全部走 Service 层（ConfigService / SQLiteStore / ...），不在 view 里重写
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time as _time
from pathlib import Path

from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from phm.algorithm._registry import MODEL_REGISTRY, get_model_entry
from phm.services.theme_service import get_theme

from . import services_bridge
from .models import AlertRecord

logger = logging.getLogger(__name__)


# ── 通用工具 ─────────────────────────────────────────────────────────────────
def _container_or_error(request):
    """获取 Container，未就绪时返回 (None, error_context_dict)。

    后台页面与 API 不同：不返回 503，而是渲染一个友好的"初始化中"提示页，
    让管理员看到系统状态而不是干等。但仍给一个标志供调用方判断。
    """
    state = services_bridge.get_state()
    if state == 'ready':
        try:
            return services_bridge.get_container(), None
        except RuntimeError as e:
            return None, {'phm_state': state, 'phm_error': str(e)}
    err = services_bridge.get_init_error() if state == 'failed' else None
    return None, {'phm_state': state, 'phm_error': err}


def _render_state_page(request, state_ctx, page_title):
    """Container 未就绪时渲染的占位页。"""
    state = state_ctx.get('phm_state', 'unknown')
    err = state_ctx.get('phm_error')
    state_text = {
        'idle': 'PHM 服务尚未启动',
        'initializing': 'PHM 服务正在初始化（加载模型，约需 10-30 秒）…',
        'failed': 'PHM 服务初始化失败',
    }.get(state, f'PHM 服务状态：{state}')
    return render(request, 'phm_site/admin/_state.html', {
        'page_title': page_title,
        'state': state,
        'state_text': state_text,
        'error': err,
    })


def _require_superuser(request):
    """超管校验。返回 (ok, error_response)。"""
    if not request.user.is_authenticated or not request.user.is_superuser:
        return False, JsonResponse(
            {'status': 'error', 'message': '仅管理员可执行此操作'},
            status=403,
        )
    return True, None


# ════════════════════════════════════════════════════════════════════════════
# 第 1 页：模型管理（纯只读卡片）
# ════════════════════════════════════════════════════════════════════════════
# 需求书 §后台「模型管理」：
#   异常检查模型和预测模型、其他专属模型卡片。显示信息、正在被哪些传感器使用。
#   不能在网页上新增和删除，需要修改配置文件。不支持 enable / disable / reload 操作。
#
# 数据源：
#   - phm.algorithm._registry.MODEL_REGISTRY（纯元数据，不 import torch）
#   - 设备树 description 里的 @ 命令（@异常检测模型 / @预测模型 / @rul:xxx）
#   - 本地资产存在性检查（HF cache snapshot / RUL 权重文件）

# 模型 key → 中文角色名
_KIND_LABEL = {
    'detector': '异常检测',
    'forecaster': '趋势预测',
    'rul': '退化预测(RUL)',
}

# @ 命令到模型 key 的映射（需求书 §补充说明：传感器可 @异常检测模型 / @预测模型 / @专属模型）
# 支持的 @ 命令前缀 → registry key。扫描 description 时按前缀匹配。
_AT_COMMAND_MAP = {
    '@tspulse': 'tspulse',
    '@异常检测模型': 'tspulse',
    '@ttm': 'ttm_r3',
    '@预测模型': 'ttm_r3',
    '@rul': 'rul',
}


def _scan_sensor_model_usage(device_tree):
    """扫描设备树 description 里的 @ 命令，返回 {model_key: [sensor_name, ...]}。"""
    usage = {}
    def walk(nodes):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if n.get('type') == 'sensor':
                desc = n.get('description') or ''
                name = n.get('name') or n.get('channelName') or n.get('sourceId') or '?'
                for cmd, mkey in _AT_COMMAND_MAP.items():
                    if cmd in desc:
                        usage.setdefault(mkey, [])
                        if name not in usage[mkey]:
                            usage[mkey].append(name)
            children = n.get('children')
            if children:
                walk(children)
    walk(device_tree or [])
    return usage


def _check_local_assets(model_key):
    """检查模型本地资产是否存在（不 import torch）。

    返回 {'available': bool, 'path': str, 'note': str}。
    HF cache 路径由 ``_hf_cache.resolve_local_model_path`` 解析（内部读
    ``HF_HOME`` 环境变量，settings.py 已 ``setdefault``），这里只做存在性检查。
    """
    entry = get_model_entry(model_key)
    if entry is None:
        return {'available': False, 'path': '', 'note': '未知模型'}

    if entry.hub_id:
        # HF 模型：检查 .hf_cache 下是否有 snapshot 目录
        # resolve_local_model_path 会处理路径解析，这里只做存在性检查
        try:
            from phm.algorithm._hf_cache import resolve_local_model_path
            local = resolve_local_model_path(entry.hub_id)
            if local and os.path.isdir(local):
                return {'available': True, 'path': local,
                        'note': f'本地快照（HF cache）'}
            return {'available': False, 'path': entry.hub_id,
                    'note': f'HF 未缓存（首次加载将联网下载）'}
        except Exception as e:
            return {'available': False, 'path': entry.hub_id,
                    'note': f'资产检查失败：{e}'}

    # 本地权重模型（RUL）：检查 models/rul/ 下权重文件
    if model_key == 'rul':
        # src/ground/models/rul/
        here = Path(__file__).resolve()
        ground_dir = here.parent.parent.parent  # src/ground/
        rul_dir = ground_dir / 'models' / 'rul'
        if rul_dir.is_dir():
            pt_files = sorted(rul_dir.glob('*.pt'))
            if pt_files:
                names = [f.name for f in pt_files]
                return {'available': True, 'path': str(rul_dir),
                        'note': f'权重：{", ".join(names)}'}
            return {'available': False, 'path': str(rul_dir), 'note': '目录存在但无 .pt 权重'}
        return {'available': False, 'path': str(rul_dir), 'note': '权重目录不存在'}

    return {'available': False, 'path': '', 'note': '未配置资产检查'}


@staff_member_required
def models_view(request):
    """模型管理页（只读卡片）。

    展示 MODEL_REGISTRY 中每个模型的元数据 + 本地资产状态 + 被哪些传感器引用。
    """
    # 设备树 usage 扫描（Container 未就绪也能跑，直接读 JSON）
    device_tree = []
    config_data = {}
    c, err = _container_or_error(request)
    if c is not None:
        try:
            config_data = c.config.load()
            device_tree = config_data.get('device_tree', [])
        except Exception as e:
            logger.warning("读取设备树失败: %s", e)
    usage = _scan_sensor_model_usage(device_tree)

    # 默认使用情况：未显式 @ 命令的传感器走系统默认（detector→tspulse, forecaster→ttm_r3）
    default_usage = _scan_default_usage(device_tree)

    cards = []
    for key, entry in MODEL_REGISTRY.items():
        assets = _check_local_assets(key)
        # 合并显式 @ 使用 + 默认使用
        sensors = list(usage.get(key, []))
        for s in default_usage.get(key, []):
            if s not in sensors:
                sensors.append(s + '（默认）')
        cards.append({
            'key': key,
            'kind': entry.kind,
            'kind_label': _KIND_LABEL.get(entry.kind, entry.kind),
            'hub_id': entry.hub_id or '（本地权重）',
            'context_length': entry.context_length,
            'prediction_length': entry.prediction_length,
            'notes': entry.notes,
            'assets_available': assets['available'],
            'assets_path': assets['path'],
            'assets_note': assets['note'],
            'sensors': sensors,
            'sensor_count': len(sensors),
        })

    return render(request, 'phm_site/admin/models.html', {
        'page_title': '模型管理',
        'cards': cards,
        'is_readonly_note': '模型为系统级配置，不支持网页新增/删除/启停，需修改配置文件',
    })


def _scan_default_usage(device_tree):
    """无显式 @ 命令的传感器：普通传感器默认用 tspulse + ttm_r3，
    特殊传感器（isSpecial）默认用 rul。返回 {model_key: [sensor_name]}。"""
    usage = {'tspulse': [], 'ttm_r3': [], 'rul': []}
    def walk(nodes):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if n.get('type') == 'sensor':
                name = n.get('name') or n.get('channelName') or '?'
                desc = n.get('description') or ''
                is_special = n.get('isSpecial') or '@rul' in desc
                if is_special:
                    if name not in usage['rul']:
                        usage['rul'].append(name)
                else:
                    # 普通传感器：没有显式 @ 命令才计入默认
                    has_explicit = any(cmd in desc for cmd in _AT_COMMAND_MAP)
                    if not has_explicit:
                        if name not in usage['tspulse']:
                            usage['tspulse'].append(name)
                        if name not in usage['ttm_r3']:
                            usage['ttm_r3'].append(name)
            children = n.get('children')
            if children:
                walk(children)
    walk(device_tree or [])
    return usage


# ════════════════════════════════════════════════════════════════════════════
# 第 2 页：仪表盘（banner 健康度 + 三卡片 + 告警趋势柱状图 + 时间窗切换）
# ════════════════════════════════════════════════════════════════════════════
# 需求书 §后台「仪表盘」：
#   头部 banner 显示系统整体健康度；中部显示三张卡片——已人工诊断的告警（含预警）、
#   LLM 诊断的告警、未诊断的告警数量；下方显示告警趋势柱状图（按单位时间分桶）。
#   三张卡片和柱状图都可以切换显示今天、最近 7 天、最近 30 天。
#
# 数据源：
#   - Container.health.system_health() → banner 健康度（[0,1]，显示时 ×100）
#   - AlertRecord ORM（db_table=alert_records，与 SQLiteStore 同表）
#     时间窗过滤 + Python 端三分类 + 桶分配聚合
#
# 设计要点：
#   - 时间窗切换走 SSR GET 参数（?window=today|7d|30d），不做 AJAX —— 需求书
#     明确"界面不会实时更新，需要刷新网页"，与告警管理页一致
#   - 柱状图用纯 CSS bar，不引 ECharts（前台大屏已有，后台保持轻量）
#   - 聚合全部在 Python 端完成，Service 层零改动（SQLiteStore 已稳定）
#   - 健康度分档阈值暂硬编码（可外置到 system_config.json，按需扩展）

# 时间窗选项（GET 参数 window 的合法值）
_WINDOW_CHOICES = ('today', '7d', '30d')
_WINDOW_DEFAULT = 'today'

# 时间窗 tab 配置（key → 中文标签）
_WINDOW_TABS = (
    ('today', '今天'),
    ('7d', '最近 7 天'),
    ('30d', '最近 30 天'),
)

# 健康度分档（threshold, tier_key, tier_text）——与 admin.css 的
# .phm-dash-banner-{tier} 配色对应。阈值暂硬编码；后续若要在线可调，
# 可外置到 system_config.json 的 dashboard 段（见 SystemConfigService 范式）。
_HEALTH_TIERS = (
    (0.80, 'normal',  '系统正常'),
    (0.50, 'warning', '存在告警'),
    (0.00, 'danger',  '健康度低'),
)

# 自动刷新间隔（秒）。用户在页面上勾选"每 Ns 自动刷新"后，URL 带 ?auto=1，
# 前端 setInterval 倒计时 reload 页面。默认关闭（?auto=1 才启用）。
# 间隔暂硬编码，后续可外置到 system_config.json。
_DASHBOARD_REFRESH_SECONDS = 15


def _health_tier(system_value):
    """将 [0,1] 健康度映射到 banner 状态分档。

    返回 (tier_key, tier_text)。tier_key 用于 CSS 类名（normal/warning/danger）。
    """
    for threshold, key, text in _HEALTH_TIERS:
        if system_value >= threshold:
            return key, text
    return _HEALTH_TIERS[-1][1], _HEALTH_TIERS[-1][2]


def _window_bounds(window, now=None):
    """计算时间窗的 [start_ts, end_ts] 与桶配置。

    返回 (start_ts, end_ts, bucket_kind, bucket_count)：
      - today: 今日 00:00 至 now，按小时分桶（24 桶）
      - 7d:    今日 00:00 往前推 6 天至 now，按天分桶（7 桶，含今日）
      - 30d:   今日 00:00 往前推 29 天至 now，按天分桶（30 桶，含今日）

    未知 window 值走 today 分支（兜底）。
    """
    if now is None:
        now = _time.time()
    now_dt = _dt.datetime.fromtimestamp(now)
    today_start = _dt.datetime(now_dt.year, now_dt.month, now_dt.day)
    if window == '7d':
        start_dt = today_start - _dt.timedelta(days=6)
        bucket_kind, bucket_count = 'day', 7
    elif window == '30d':
        start_dt = today_start - _dt.timedelta(days=29)
        bucket_kind, bucket_count = 'day', 30
    else:  # 'today' 或未知值兜底
        start_dt = today_start
        bucket_kind, bucket_count = 'hour', 24
    return start_dt.timestamp(), now, bucket_kind, bucket_count


def _classify_verdict(human_v, llm_v):
    """三分类告警诊断状态。

    返回 'human' / 'llm' / 'undiagnosed'：
      - human_verdict 非空（'real'/'false_alarm'/'uncertain'）→ 'human'
      - 否则 llm_verdict 非空 → 'llm'
      - 都空（None 或 ''）→ 'undiagnosed'

    空字符串与 None 都算"未标注"（VERDICT_CHOICES 第一项是 ''）。
    优先级与 ``AlertRecord.final_status`` 一致：人工 > LLM。
    """
    if human_v:
        return 'human'
    if llm_v:
        return 'llm'
    return 'undiagnosed'


def _bucket_index(ts, start_ts, bucket_kind):
    """计算时间戳 ts 落在第几桶（从 0 开始）。

    bucket_kind='hour' → 3600 秒/桶；'day' → 86400 秒/桶。
    """
    span = 3600 if bucket_kind == 'hour' else 86400
    return int((ts - start_ts) // span)


def _format_bucket_label(idx, bucket_kind, start_ts):
    """桶 idx 的展示标签（短形式，用于 x 轴刻度）。

    - hour：纯小时数字，如 '14'（24 桶紧凑显示，悬停 title 仍给完整信息）
    - day：纯日数字，如 '21'（7/30 桶也紧凑）
    """
    span = 3600 if bucket_kind == 'hour' else 86400
    bucket_start = start_ts + idx * span
    dt = _dt.datetime.fromtimestamp(bucket_start)
    return str(dt.hour) if bucket_kind == 'hour' else str(dt.day)


def _format_bucket_title(idx, bucket_kind, start_ts):
    """桶 idx 的鼠标悬停 title（完整信息，无截断）。

    - hour：'2026-07-21 14:00'
    - day：'2026-07-21'
    """
    span = 3600 if bucket_kind == 'hour' else 86400
    bucket_start = start_ts + idx * span
    dt = _dt.datetime.fromtimestamp(bucket_start)
    return dt.strftime('%Y-%m-%d %H:00') if bucket_kind == 'hour' else dt.strftime('%Y-%m-%d')


def _collect_dashboard_metrics(window, alerts):
    """聚合仪表盘统计指标。

    alerts: 可迭代对象，每项需有 created_at/human_verdict/llm_verdict 字段
            （AlertRecord ORM 或鸭子类型）。
    返回 {
        window, start_ts, end_ts, bucket_kind,
        counts: {human, llm, undiagnosed, total},
        breakdown: {   # 三分类下各自的 verdict 细分（real/false_alarm/uncertain）
            human:       {real, false_alarm, uncertain},
            llm:         {real, false_alarm, uncertain},
        },
        buckets: [   # 含 0 桶，按桶序排列
            {
                label, title, count,           # count = 该桶总数
                parts: {                       # 该桶按来源×verdict 细分（供前端 stacked bar）
                    human: {real, false_alarm, uncertain},
                    llm:   {real, false_alarm, uncertain},
                    undiagnosed: <int>,
                },
            }, ...
        ],
    }
    """
    start_ts, end_ts, bucket_kind, bucket_count = _window_bounds(window)
    counts = {'human': 0, 'llm': 0, 'undiagnosed': 0, 'total': 0}
    breakdown = {
        'human': {'real': 0, 'false_alarm': 0, 'uncertain': 0},
        'llm':   {'real': 0, 'false_alarm': 0, 'uncertain': 0},
    }
    bucket_counts = [0] * bucket_count
    # 每桶的来源×verdict 矩阵（stacked bar 数据源）
    bucket_parts = [
        {
            'human': {'real': 0, 'false_alarm': 0, 'uncertain': 0},
            'llm':   {'real': 0, 'false_alarm': 0, 'uncertain': 0},
            'undiagnosed': 0,
        }
        for _ in range(bucket_count)
    ]
    for a in alerts:
        ts = a.created_at
        # 时间窗外丢弃（ORM 已过滤；防御：mock/历史调用可能传入越界数据）
        if ts < start_ts or ts > end_ts:
            continue
        category = _classify_verdict(a.human_verdict, a.llm_verdict)
        counts[category] += 1
        counts['total'] += 1
        # verdict 细分：human 类用 human_verdict，llm 类用 llm_verdict
        verdict_value = None
        if category in ('human', 'llm'):
            verdict_value = a.human_verdict if category == 'human' else a.llm_verdict
            if verdict_value in breakdown[category]:
                breakdown[category][verdict_value] += 1
        idx = _bucket_index(ts, start_ts, bucket_kind)
        if 0 <= idx < bucket_count:
            bucket_counts[idx] += 1
            parts = bucket_parts[idx]
            if category == 'undiagnosed':
                parts['undiagnosed'] += 1
            elif verdict_value in parts[category]:
                parts[category][verdict_value] += 1
    buckets = [
        {
            'label': _format_bucket_label(i, bucket_kind, start_ts),
            'title': _format_bucket_title(i, bucket_kind, start_ts),
            'count': bucket_counts[i],
            'parts': bucket_parts[i],
        }
        for i in range(bucket_count)
    ]
    return {
        'window': window,
        'start_ts': start_ts,
        'end_ts': end_ts,
        'bucket_kind': bucket_kind,
        'counts': counts,
        'breakdown': breakdown,
        'buckets': buckets,
    }


@staff_member_required
def dashboard_view(request):
    """仪表盘页。

    GET ?window=today|7d|30d 切换时间窗（默认 today，非法值兜底）。
    Container 未就绪时渲染占位页（_state.html），不 500。
    """
    window = request.GET.get('window', _WINDOW_DEFAULT)
    if window not in _WINDOW_CHOICES:
        window = _WINDOW_DEFAULT

    # 自动刷新：URL 带 ?auto=1 启用（默认关闭，勾选 checkbox 激活）
    auto_refresh = request.GET.get('auto') == '1'

    # Container 三态守门
    c, err = _container_or_error(request)
    if c is None:
        return _render_state_page(request, err, '仪表盘')

    # 健康度（[0,1] → 显示时 ×100）
    try:
        health_data = c.health.system_health()
    except Exception as e:
        logger.warning("system_health failed: %s", e)
        health_data = {'system': 1.0, 'channels': {}, 'threshold': 0}

    system_value = float(health_data.get('system', 1.0))
    tier_key, tier_text = _health_tier(system_value)

    # 天地链路状态（与前台顶栏 system_info_view 同源：services_bridge.get_link_status）
    # link_status: {rtt_ms, status, last_success_ts}。status: online/degraded/offline/waiting
    try:
        link = services_bridge.get_link_status()
        latency_ms = round(link['rtt_ms'], 1) if link.get('rtt_ms') is not None else None
        link_status = link.get('status', 'unknown')
    except Exception as e:
        logger.warning("get_link_status failed: %s", e)
        latency_ms = None
        link_status = 'unknown'

    banner = {
        'system_pct': round(system_value * 100, 1),
        'tier': tier_key,
        'tier_text': tier_text,
        'channel_count': len(health_data.get('channels', {})),
        'link_latency_ms': latency_ms,
        'link_status': link_status,
    }

    # 告警时间窗聚合
    start_ts, _end_ts, _bucket_kind, _bucket_count = _window_bounds(window)
    try:
        alerts_qs = AlertRecord.objects.filter(
            is_deleted=0,
            created_at__gte=start_ts,
            created_at__lte=_end_ts,
        ).only('created_at', 'human_verdict', 'llm_verdict')
        metrics = _collect_dashboard_metrics(window, alerts_qs)
    except Exception as e:
        logger.warning("dashboard alerts query failed: %s", e)
        metrics = _collect_dashboard_metrics(window, [])

    # 时间窗 tabs（active 标记当前选中）
    tabs = [
        {'key': k, 'label': lbl, 'active': (k == window)}
        for k, lbl in _WINDOW_TABS
    ]

    # 柱状图最大值（用于 CSS 高度比例；为 0 时模板走空状态）
    # max_bucket = 单桶总数最大值（前端按段比例渲染 stacked bar）
    max_bucket = max((b['count'] for b in metrics['buckets']), default=0)

    # 把 buckets 序列化为前端 JS 可用的结构（parts 矩阵 + label/title/count）
    # 模板渲染时 SSR 输出一份 JSON 给 JS 用，避免 JS 再 fetch
    buckets_json = json.dumps(metrics['buckets'], ensure_ascii=False)

    return render(request, 'phm_site/admin/dashboard.html', {
        'page_title': '仪表盘',
        'banner': banner,
        'metrics': metrics,
        'tabs': tabs,
        'max_bucket': max_bucket,
        'buckets_json': buckets_json,
        'auto_refresh': auto_refresh,
        'refresh_seconds': _DASHBOARD_REFRESH_SECONDS,
    })
