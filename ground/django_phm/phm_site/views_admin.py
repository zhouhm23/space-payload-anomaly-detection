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
import math
import os
import time as _time
from pathlib import Path

from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
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
# 部署位置标签（天基预留 OTA，地基本地推理）
_DEPLOY_LABEL = {
    'ground': '地基',
    'space': '天基',
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
            'deploy': entry.deploy,
            'deploy_label': _DEPLOY_LABEL.get(entry.deploy, entry.deploy),
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

# 自动刷新间隔（秒）。dashboard 默认开启自动刷新（URL 不带 auto 也启用），
# 用户可在页面勾选框关闭（写 ?auto=0）。前端 setInterval 倒计时 reload 页面。
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

    # 自动刷新：默认开启（?auto=0 才关闭）。需求书反馈：dashboard 默认自动刷新。
    auto_refresh = request.GET.get('auto', '1') != '0'

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


# ════════════════════════════════════════════════════════════════════════════
# 第 3 页：回收站（仅超管可改）
# ════════════════════════════════════════════════════════════════════════════
# 需求书 §后台「回收站（仅管理员可修改）」：
#   和告警管理列表形态一致，但功能栏只有「永久删除」+「恢复」两个按钮。
#
# 数据源：SQLiteStore 三张业务表的 is_deleted=1 行（detection_results /
#   alert_records / diagnosis_records）。第 1 节公共前置已加 query_deleted /
#   restore / purge_by_ids 三个方法，本 view 是薄壳。

# URL ?table= 白名单（key → (SQLiteStore 表名, 中文标签)）
_RECYCLE_TABLE_MAP = {
    'alerts':     ('alert_records',     '告警记录'),
    'detections': ('detection_results', '检测明细'),
    'diagnoses':  ('diagnosis_records', '诊断记录'),
}
_RECYCLE_TABLE_DEFAULT = 'alerts'
_RECYCLE_LIMIT_DEFAULT = 20
_RECYCLE_LIMIT_MAX = 1000
_RECYCLE_PAGE_SIZE_OPTIONS = [20, 50, 100, 200]


def _parse_recycle_table(key):
    """解析 ?table= 参数，返回 (table_key, sql_table, label)。

    非法值兜底为 alerts（默认 tab）。返回三元组供 view 与模板共用。
    """
    if key not in _RECYCLE_TABLE_MAP:
        key = _RECYCLE_TABLE_DEFAULT
    sql_table, label = _RECYCLE_TABLE_MAP[key]
    return key, sql_table, label


def _parse_recycle_limit(raw):
    """解析 ?limit= 参数，限幅 [1, 1000]，非法值兜底默认。"""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _RECYCLE_LIMIT_DEFAULT
    return max(1, min(n, _RECYCLE_LIMIT_MAX))


def _parse_id_list(raw):
    """把请求里的 ids（list 或逗号串）规整成 list[int]。

    支持 JSON 数组 / 逗号分隔字符串 / 单个 id。
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for v in raw:
            try:
                iv = int(v)
                if iv > 0:
                    out.append(iv)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, int):
        return [raw] if raw > 0 else []
    if isinstance(raw, str):
        out = []
        for part in raw.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                iv = int(part)
                if iv > 0:
                    out.append(iv)
            except ValueError:
                continue
        return out
    return []


def _verdict_badge(verdict):
    """verdict → CSS 徽章类（与仪表盘色序一致：实警红/虚警绿/待定黄）。"""
    return {
        'real':        'phm-badge-red',
        'false_alarm': 'phm-badge-green',
        'uncertain':   'phm-badge-yellow',
    }.get(verdict, 'phm-badge-gray')


def _alert_type_badge(alert_type):
    """alert_type → CSS 徽章类（实测红/预测黄/联合紫）。"""
    # 注：联合告警 (joint) 用 cyan 便于与双色彩区分
    return {
        'measured':  'phm-badge-red',
        'predicted': 'phm-badge-yellow',
        'joint':     'phm-badge-cyan',
    }.get(alert_type, 'phm-badge-gray')


def _build_sensor_meta(config_service):
    """从 device_tree 构造 {channelName: {sensor_name, unit}} 映射。

    需求书「告警和预警管理」要求列里有「传感器名称」（区别于通道名 channelName，
    在 device_tree 中是 sensor.name 字段，通常与 channelName 相同但可独立配置）
    和「遥测值+单位」展示。回收站列表复用告警管理的列定义，所以也要这两个字段。
    """
    mapping = {}
    if config_service is None:
        return mapping
    try:
        body = config_service.load()
        tree = body.get('device_tree') or []
    except Exception as e:
        logger.warning("recycle: load device_tree failed: %s", e)
        return mapping

    def walk(nodes):
        for n in nodes or []:
            if not isinstance(n, dict):
                continue
            if n.get('type') == 'sensor':
                ch = n.get('channelName') or n.get('sourceId') or n.get('name')
                if ch and ch not in mapping:
                    mapping[ch] = {
                        'sensor_name': n.get('name') or ch,
                        'unit': n.get('unit') or '',
                    }
            walk(n.get('children'))
    walk(tree)
    return mapping


def _final_status_badge(final_status):
    """综合状态 final_status → CSS 徽章类。

    final_status 优先级 human > llm > 核验（active/pending/confirmed/false）。
    real→红 / false_alarm→绿 / uncertain→黄 / confirmed→蓝 / false→绿 /
    pending→黄 / active→灰。
    """
    return {
        'real':        'phm-badge-red',
        'false_alarm': 'phm-badge-green',
        'uncertain':   'phm-badge-yellow',
        'confirmed':   'phm-badge-blue',
        'false':       'phm-badge-green',
        'pending':     'phm-badge-yellow',
        'active':      'phm-badge-gray',
    }.get(final_status, 'phm-badge-gray')


# ── 中文 label 映射（单一真相源，alert_view + recycle_view 共用） ──────
# 解决中英混合问题：数据行直接输出数据库英文原始值（measured/real/active），
# 筛选栏用中文，JS 局部刷新又是中文——SSR 与 JS 不一致。这里统一由后端
# 提供 *_label 字段，模板和 JS 都用 label，保持一致。
_ALERT_TYPE_LABEL = {
    'measured':  '实测告警',
    'predicted': '预测预警',
    'joint':     '联合告警',
}
_VERDICT_LABEL = {
    'real':        '实警',
    'false_alarm': '虚警',
    'uncertain':   '待定',
}
# 综合状态/告警状态中文映射（final_status + status 共用）
_STATUS_LABEL = {
    'active':      '活跃',
    'real':        '实警',
    'false_alarm': '虚警',
    'uncertain':   '待定',
    'confirmed':   '已确认',
    'false':       '误报',
    'pending':     '待处理',
}


def _label(mapping, value):
    """从 mapping 取中文 label，未命中时回退原值（None 回退 '—'）。"""
    if not value:
        return '—'
    return mapping.get(value, value)


@staff_member_required
def recycle_view(request):
    """回收站页（GET）。

    GET ?table=alerts|detections|diagnoses 切换三张资源（默认 alerts）。
    GET ?limit=N 控制单页条数（1-1000，默认 200）。
    Container 未就绪时渲染占位页（_state.html），不 500。

    列定义对齐需求书 §后台「告警和预警管理」（10 列）减去「操作」列
    （回收站无抽屉操作，功能栏统一为「恢复」+「永久删除」），加上「删除时间」
    （回收站特有）。即：复选框 / id / 类型 / 传感器名 / 遥测值 / 异常分数 /
    告警时间 / LLM 状态 / 人工状态 / 综合状态 / 删除时间。
    """
    table_key, sql_table, label = _parse_recycle_table(request.GET.get('table'))
    limit = _parse_recycle_limit(request.GET.get('limit'))
    page = _parse_alert_page(request.GET.get('page'))  # 复用通用 page 解析

    c, err = _container_or_error(request)
    if c is None:
        return _render_state_page(request, err, '回收站')

    # 总行数（分页用）
    try:
        total_count = c.sqlite.count_deleted(sql_table)
    except Exception as e:
        logger.warning("recycle count_deleted(%s) failed: %s", sql_table, e)
        total_count = 0

    # 计算分页：page 超出 total_pages 时兜底到最后一页
    total_pages = max(1, math.ceil(total_count / limit)) if total_count > 0 else 1
    if page > total_pages:
        page = total_pages
    offset = (page - 1) * limit

    # 调 SQLiteStore.query_deleted（按 deleted_at DESC 排序）
    try:
        rows = c.sqlite.query_deleted(sql_table, limit=limit, offset=offset)
    except Exception as e:
        logger.warning("recycle query_deleted(%s) failed: %s", sql_table, e)
        rows = []

    # 传感器元信息（告警列表用：传感器名 + 单位）
    sensor_meta = _build_sensor_meta(getattr(c, 'config', None))

    # 给每行加徽章类（模板直接用）
    decorated = []
    for r in rows:
        item = dict(r)
        item['alert_type_badge'] = _alert_type_badge(r.get('alert_type'))
        item['llm_verdict_badge'] = _verdict_badge(r.get('llm_verdict'))
        item['human_verdict_badge'] = _verdict_badge(r.get('human_verdict'))
        item['final_status_badge'] = _final_status_badge(r.get('final_status'))
        # 中文 label（与筛选栏一致，避免数据行显示英文原始值）
        item['alert_type_label'] = _label(_ALERT_TYPE_LABEL, r.get('alert_type'))
        item['llm_verdict_label'] = _label(_VERDICT_LABEL, r.get('llm_verdict')) if r.get('llm_verdict') else '未诊断'
        item['human_verdict_label'] = _label(_VERDICT_LABEL, r.get('human_verdict')) if r.get('human_verdict') else '未标注'
        item['final_status_label'] = _label(_STATUS_LABEL, r.get('final_status'))
        # 传感器名 + 单位（告警列表列用）
        meta = sensor_meta.get(r.get('channel'))
        item['sensor_name'] = meta['sensor_name'] if meta else (r.get('channel') or '—')
        item['unit'] = meta['unit'] if meta else ''
        decorated.append(item)

    # 三个 tab（active 标记当前选中）
    tabs = [
        {'key': k, 'label': lbl[1], 'active': (k == table_key)}
        for k, lbl in _RECYCLE_TABLE_MAP.items()
    ]

    # 是否超管（模板依据此显示/隐藏批量操作按钮）
    is_superuser = request.user.is_authenticated and request.user.is_superuser

    return render(request, 'phm_site/admin/recycle.html', {
        'page_title': '回收站',
        'table_key': table_key,
        'table_label': label,
        'rows': decorated,
        'tabs': tabs,
        'limit': limit,
        'is_superuser': is_superuser,
        'csrf_token_str': request.META.get('CSRF_COOKIE', ''),
        # 分页（与 alert_view 同款，复用 _pagination.html / _page_size_select.html）
        'total_count': total_count,
        'page': page,
        'total_pages': total_pages,
        'page_range': _build_page_range(page, total_pages),
        'current_filters': {'table': table_key if table_key != _RECYCLE_TABLE_DEFAULT else ''},
        'page_size_options': _RECYCLE_PAGE_SIZE_OPTIONS,
    })


@staff_member_required
@require_http_methods(['POST'])
def recycle_restore_api(request):
    """恢复软删记录（POST，仅超管）。

    入参 JSON: {table: 'alerts|detections|diagnoses', ids: [int,...]}
    出参 JSON: {status: 'ok', restored: N} 或 {status: 'error', message}
    """
    ok, err_resp = _require_superuser(request)
    if not ok:
        return err_resp
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '请求体不是合法 JSON'},
                            status=400)
    table_key, sql_table, _label = _parse_recycle_table(body.get('table'))
    ids = _parse_id_list(body.get('ids'))
    if not ids:
        return JsonResponse({'status': 'error', 'message': '未提供有效 id'},
                            status=400)

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(
                                 err.get('phm_state'))}, status=503)
    try:
        n = c.sqlite.restore(sql_table, ids)
    except Exception as e:
        logger.warning("recycle restore failed: %s", e)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'ok', 'restored': n, 'table': table_key})


@staff_member_required
@require_http_methods(['POST'])
def recycle_purge_api(request):
    """永久删除（物理清除）软删记录（POST，仅超管）。

    入参 JSON: {table: 'alerts|detections|diagnoses', ids: [int,...]}
    出参 JSON: {status: 'ok', purged: N} 或 {status: 'error', message}

    安全约束：purge_by_ids 只删 is_deleted=1 的行，活跃数据不受影响。
    """
    ok, err_resp = _require_superuser(request)
    if not ok:
        return err_resp
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '请求体不是合法 JSON'},
                            status=400)
    table_key, sql_table, _label = _parse_recycle_table(body.get('table'))
    ids = _parse_id_list(body.get('ids'))
    if not ids:
        return JsonResponse({'status': 'error', 'message': '未提供有效 id'},
                            status=400)

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(
                                 err.get('phm_state'))}, status=503)
    try:
        n = c.sqlite.purge_by_ids(sql_table, ids)
    except Exception as e:
        logger.warning("recycle purge failed: %s", e)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'ok', 'purged': n, 'table': table_key})


# ════════════════════════════════════════════════════════════════════════════
# 第 9 页：用户管理 + 审计日志（SimpleUI 默认 + 权限说明静态页）
# ════════════════════════════════════════════════════════════════════════════
# 需求书 §后台：
#   - 「用户管理（simpleui默认），加个说明按钮，打开显示权限说明面板」
#   - 「审计日志（simpleui默认）」
#
# 实现策略：
#   - 用户管理（User/Group CRUD）：完全走 SimpleUI 默认（django.contrib.auth 已注册）
#   - 审计日志（LogEntry 浏览）：完全走 SimpleUI 默认（django.contrib.admin 已注册）
#     范围已确认：Django LogEntry 仅记录 admin 站内 ModelAdmin 增删改，
#     不覆盖自定义页 AJAX 操作（用户已确认这是接受的边界）
#   - 唯一新增：权限说明静态页 /admin/phm_site/permissions/

# 角色清单（与 _require_superuser / @staff_member_required 的判定对齐）
_PERMISSION_ROLES = [
    {
        'key': 'anonymous',
        'name': '匿名用户',
        'desc': '未登录的访客。仅能看到登录页，无法访问任何后台功能。',
        'badge': 'phm-badge-gray',
    },
    {
        'key': 'staff',
        'name': '普通管理员（staff）',
        'desc': '已登录但非超级管理员。可读所有页面，可执行只读/可逆操作。',
        'badge': 'phm-badge-blue',
    },
    {
        'key': 'superuser',
        'name': '超级管理员（superuser）',
        'desc': '拥有全部权限，包括所有写操作、用户管理、系统配置。',
        'badge': 'phm-badge-red',
    },
]

# 各功能页的权限矩阵：{操作: {role: '✓' / '只读' / '—'}}
# 与 _require_superuser helper + @staff_member_required 的实际判定对齐
_PERMISSION_MATRIX = [
    {
        'page': '仪表盘',
        'url': '/admin/phm_site/dashboard/',
        'anonymous': '—', 'staff': '✓ 读', 'superuser': '✓ 读',
    },
    {
        'page': '告警与预警管理',
        'url': '/admin/phm_site/alert/',
        'anonymous': '—', 'staff': '✓ 读 + 标注 + LLM 诊断 + 导出', 'superuser': '✓ 全部 + 新增 + 移到回收站',
    },
    {
        'page': '回收站',
        'url': '/admin/phm_site/recycle/',
        'anonymous': '—', 'staff': '✓ 只读列表', 'superuser': '✓ 恢复 + 永久删除',
    },
    {
        'page': '设备树管理',
        'url': '/admin/phm_site/device-tree/',
        'anonymous': '—', 'staff': '✓ 只读', 'superuser': '✓ 新建 + 编辑 + 拖拽 + 删除',
    },
    {
        'page': '系统设置',
        'url': '/admin/phm_site/settings/',
        'anonymous': '—', 'staff': '✓ 只读', 'superuser': '✓ 修改系统配置 + 前台主题（通道校准只读）',
    },
    {
        'page': '模型管理',
        'url': '/admin/phm_site/models/',
        'anonymous': '—', 'staff': '✓ 只读', 'superuser': '✓ 只读（修改需改配置文件 + 重启）',
    },
    {
        'page': '用户与组管理',
        'url': '/admin/auth/user/',
        'anonymous': '—', 'staff': '—', 'superuser': '✓ 全部（SimpleUI 默认）',
    },
    {
        'page': '审计日志',
        'url': '/admin/admin/logentry/',
        'anonymous': '—', 'staff': '✓ 只读', 'superuser': '✓ 只读',
    },
]

# 审计日志范围说明（用户已确认接受的边界）
_AUDIT_SCOPE_NOTES = [
    'Django LogEntry 默认仅记录 admin 站内 ModelAdmin 的增删改操作（用户/组/业务模型列表页）。',
    '自定义页的 AJAX 写操作（如回收站恢复/永久删除、告警标注、系统设置保存、设备树保存）当前<strong>不</strong>写入 LogEntry。',
    'CLI 命令（manage.py phm_*）与 API 调用（/api/v2/*）也<strong>不</strong>计入 LogEntry。',
    '如需扩展到自定义页操作，需在各自定义页 view 里手动 log_action()（未来工作）。',
]


@staff_member_required
def permissions_view(request):
    """权限说明静态页（GET）。

    纯 SSR，不依赖 Container。展示三大角色（匿名/staff/superuser）的权限矩阵，
    帮助管理员快速了解各角色能做什么、各页面需要什么权限。
    """
    # 当前用户角色（高亮当前行）
    if not request.user.is_authenticated:
        current_role = 'anonymous'
    elif request.user.is_superuser:
        current_role = 'superuser'
    else:
        current_role = 'staff'

    return render(request, 'phm_site/admin/permissions.html', {
        'page_title': '权限说明',
        'roles': _PERMISSION_ROLES,
        'matrix': _PERMISSION_MATRIX,
        'audit_notes': _AUDIT_SCOPE_NOTES,
        'current_role': current_role,
    })


# ════════════════════════════════════════════════════════════════════════════
# 第 5 页：系统设置（系统配置 / 前台主题 / 通道校准三类）
# ════════════════════════════════════════════════════════════════════════════
# 需求书 §后台「系统设置（仅管理员可修改）」：
#   含系统配置、前台主题、通道校准三类。各种配置项，中文名称（有悬浮描述）、
#   变量名、值。
#
# 数据源：
#   - system：SystemConfigService.raw_with_docs() / save()
#   - theme：ThemeService.raw_with_docs() / save()
#   - calibration：直接读 channel_calibration.json（纯只读，离线标定产物）
#
# 设计要点：
#   - 通道校准只读：Day15 离线 LOO 标定产物，在线编辑会破坏口径
#   - llm.timeout_sec 等 .env 管理的 key 在 UI 灰显
#   - save 通过原子写 + load() 热生效，无需重启

# 三类 tab 配置（key → (label, service_kind)）
_SETTINGS_CATEGORIES = (
    ('system',      '系统配置'),
    ('theme',       '前台主题'),
    ('calibration', '通道校准（只读）'),
)
_SETTINGS_CATEGORY_DEFAULT = 'system'
_SETTINGS_CATEGORY_KEYS = frozenset(k for k, _ in _SETTINGS_CATEGORIES)

# 通道校准文件位置（与 CalibrationConfig.DEFAULT_CONFIG_PATH 同源，但直接读原文）。
# __file__ = src/ground/django_phm/phm_site/views_admin.py
# 上 3 级 → src/ground/，再拼 data/channel_calibration.json（与 SystemConfigService 同范式）。
_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "channel_calibration.json",
)


def _parse_settings_category(raw):
    """解析 ?category= 参数，非法值兜底 system。"""
    if raw not in _SETTINGS_CATEGORY_KEYS:
        return _SETTINGS_CATEGORY_DEFAULT
    return raw


def _build_settings_items(raw, display_names, *, readonly_predicate=None):
    """把 raw_with_docs() 的 section→{key:value} 扁平化成 UI 行列表。

    Args:
        raw: dict[section, dict[key, value]] —— service.raw_with_docs() 输出。
        display_names: dict[section, dict[key, str]] —— service.display_names() 输出。
        readonly_predicate: 可选回调 (section, key) -> bool，True 表示该项只读。

    返回结构：
        [
            {
                'section': 'thresholds',
                'section_label': '异常检测阈值',  # display_names[section]['_doc']
                'section_doc': '<section _doc in JSON>',  # 原文中的 section _doc（可能为空）
                'key': 'anomaly',
                'name': '异常分数阈值',          # display_names[section][key]
                'doc': '<key _doc>',            # 原文 key 的 _doc（如有）
                'value': 0.5,
                'value_kind': 'float',          # bool/int/float/str/object/array
                'editable': True,
            },
            ...
        ]
    """
    items = []
    for section, sec_values in raw.items():
        if section.startswith('_') or not isinstance(sec_values, dict):
            continue
        sec_names = display_names.get(section, {})
        section_label = sec_names.get('_doc', section)
        section_doc = sec_values.get('_doc', '') if isinstance(sec_values, dict) else ''
        # 排序：按 display_names 顺序（已定义顺序），未在 display_names 的追加在后
        ordered_keys = [k for k in sec_names.keys() if k != '_doc']
        extra_keys = [k for k in sec_values.keys()
                      if k != '_doc' and k not in ordered_keys]
        for key in ordered_keys + extra_keys:
            if key.startswith('_'):
                continue
            if key not in sec_values:
                continue  # display_names 里有但 JSON 没有的 key 跳过
            value = sec_values[key]
            value_kind = _classify_value_kind(value)
            editable = value_kind in ('bool', 'int', 'float', 'str')
            if editable and readonly_predicate and readonly_predicate(section, key):
                editable = False
            items.append({
                'section': section,
                'section_label': section_label,
                'section_doc': section_doc,
                'key': key,
                'name': sec_names.get(key, key),
                'doc': '',  # 单 key 的 _doc 暂不在 JSON 中维护（_doc 是 section 级）
                'value': value,
                'value_kind': value_kind,
                'editable': editable,
            })
    return items


def _classify_value_kind(value):
    """把 Python 值映射到 UI 类型标签（bool/int/float/str/object/array）。"""
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'float'
    if isinstance(value, str):
        return 'str'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    return 'unknown'


def _group_items_by_section(items):
    """把扁平 items 按 section 分组，便于模板按卡片渲染。

    返回 [{'section': 'thresholds', 'label': '...', 'doc': '...', 'items': [...]}, ...]
    顺序：保持首次出现顺序（dict 保留插入序）。
    """
    groups = {}
    order = []
    for it in items:
        sec = it['section']
        if sec not in groups:
            groups[sec] = {
                'section': sec,
                'label': it['section_label'],
                'doc': it['section_doc'],
                'items': [],
            }
            order.append(sec)
        groups[sec]['items'].append(it)
    return [groups[s] for s in order]


@staff_member_required
def settings_view(request):
    """系统设置页（GET）。

    GET ?category=system|theme|calibration 切换三类（默认 system）。
    Container 未就绪时也能渲染前两类（不依赖 Container；但 calibration 需要
    Container 解析的 channel 名映射，这里仍用文件直读，故三态机守门放宽：
    只在 service 异常时退回占位页）。
    """
    category = _parse_settings_category(request.GET.get('category'))
    is_superuser = request.user.is_authenticated and request.user.is_superuser

    groups = []
    save_url = None
    error_msg = None

    try:
        if category == 'system':
            from phm.services.system_config_service import get_system_config
            svc = get_system_config()
            raw = svc.raw_with_docs()
            items = _build_settings_items(
                raw, svc.display_names(),
                readonly_predicate=lambda s, k: svc.is_readonly(s, k),
            )
            groups = _group_items_by_section(items)
        elif category == 'theme':
            svc = get_theme()
            raw = svc.raw_with_docs()
            items = _build_settings_items(
                raw, svc.display_names(),
                readonly_predicate=lambda s, k: svc.is_readonly(s, k),
            )
            groups = _group_items_by_section(items)
        else:  # calibration
            groups = _build_calibration_groups()
    except Exception as e:
        logger.warning("settings_view(%s) failed: %s", category, e, exc_info=True)
        error_msg = str(e)

    # 三类 tab（active 标记当前）
    tabs = [
        {'key': k, 'label': lbl, 'active': (k == category)}
        for k, lbl in _SETTINGS_CATEGORIES
    ]

    return render(request, 'phm_site/admin/settings.html', {
        'page_title': '系统设置',
        'category': category,
        'tabs': tabs,
        'groups': groups,
        'is_superuser': is_superuser,
        'error_msg': error_msg,
        # 给 JS 用：每行可编辑的项集合（用于批量收集改动）
        'editable_json': json.dumps(
            [{'section': it['section'], 'key': it['key']}
             for grp in groups for it in grp['items'] if it['editable']],
            ensure_ascii=False,
        ),
    })


def _build_calibration_groups():
    """读 channel_calibration.json 原文，构造只读展示分组。

    每条记录是 {channel: {flip, score_type, threshold, threshold_name, ...}}，
    UI 按 channel 一行展开，列出关键字段（threshold/score_type/flip/threshold_name）。
    嵌套数组字段（freq_band_mean/std）折叠成「<N 个数值>」摘要，不展开。
    """
    if not os.path.exists(_CALIBRATION_PATH):
        return [{
            'section': 'calibration',
            'label': '通道校准',
            'doc': 'channel_calibration.json 不存在（系统未标定）',
            'items': [],
            'readonly_hint': True,
        }]
    try:
        with open(_CALIBRATION_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        logger.warning("failed to read calibration", exc_info=True)
        return [{
            'section': 'calibration',
            'label': '通道校准',
            'doc': 'channel_calibration.json 解析失败',
            'items': [],
            'readonly_hint': True,
        }]
    items = []
    for channel, cfg in raw.items():
        if channel.startswith('_') or not isinstance(cfg, dict):
            continue
        # 把每个 channel 当成一个 section，cfg 内字段当 items
        sec_items = []
        for key, value in cfg.items():
            if key.startswith('_'):
                continue
            value_kind = _classify_value_kind(value)
            # 数组字段折叠成摘要
            display_value = value
            if value_kind == 'array':
                display_value = f"<{len(value)} 个数值>"
            elif value is None:
                display_value = "—"
            sec_items.append({
                'section': channel,
                'section_label': f"通道 {channel}",
                'section_doc': '',
                'key': key,
                'name': key,  # calibration 字段名直接展示（中文映射暂无）
                'doc': '',
                'value': display_value,
                'value_kind': value_kind,
                'editable': False,  # 全只读
            })
        if sec_items:
            items.append({
                'section': channel,
                'label': f"通道 {channel}",
                'doc': '',
                'items': sec_items,
                'readonly_hint': True,
            })
    return items


@staff_member_required
@require_http_methods(['POST'])
def settings_save_api(request):
    """保存单个配置项（POST，仅超管）。

    入参 JSON: {category: 'system'|'theme', section: str, key: str, value: any}
    出参 JSON: {status: 'ok', old, new} 或 {status: 'error', message}
    calibration 类直接 403（只读）。
    """
    ok, err_resp = _require_superuser(request)
    if not ok:
        return err_resp
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '请求体不是合法 JSON'},
                            status=400)

    category = body.get('category')
    if category not in ('system', 'theme'):
        return JsonResponse({'status': 'error',
                             'message': f'category 只接受 system/theme（calibration 只读）'},
                            status=400)
    section = body.get('section')
    key = body.get('key')
    if 'value' not in body:
        return JsonResponse({'status': 'error', 'message': '缺少 value 字段'},
                            status=400)
    value = body['value']
    if not isinstance(section, str) or not isinstance(key, str):
        return JsonResponse({'status': 'error', 'message': 'section/key 必须为字符串'},
                            status=400)

    try:
        if category == 'system':
            from phm.services.system_config_service import get_system_config
            result = get_system_config().save(section, key, value)
        else:  # theme
            result = get_theme().save(section, key, value)
    except Exception as e:
        logger.warning("settings_save_api failed: %s", e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    if result.get('status') != 'ok':
        return JsonResponse(result, status=400)
    return JsonResponse(result)


# ════════════════════════════════════════════════════════════════════════════
# 第 4 页：告警和预警管理（仅 measured 告警；predicted 预警在仪表盘）
# ════════════════════════════════════════════════════════════════════════════
# 需求书 §后台「告警和预警管理（含预警）」：
#   每行显示 id、告警或预警、传感器名称、遥测值、异常分数、告警时间(UTC)、
#   llm 诊断状态、人工诊断状态、综合状态、操作（点击抽屉显示波形/描述/LLM/标注）。
#   上方：新增 / 移动到回收站 / 删除 / llm 诊断 / 人工标注 / 导出（都可批量）。
#
# 决策（已确认）：
#   - 本页只管 measured 告警（持久化在 SQLite alert_records）
#   - predicted 预警在仪表盘已展示，本页不重复
#   - 列表非实时更新（需求书：「界面不会实时更新，需要刷新网页获取新数据」）

# 筛选参数白名单与默认值（对齐需求书 §后台 L97 列定义：类型 / 传感器名称 /
# LLM 诊断 / 人工诊断 / 综合状态；外加时间窗，但不暴露数据库 status 字段）
_ALERT_FILTER_DEFAULTS = {
    'channel': None,        # str | None — 传感器 channelName（UI 标签为"传感器名称"）
    'alert_type': None,     # 'measured'|'predicted'|'joint' | None
    'llm_verdict': None,    # 'real'|'false_alarm'|'uncertain'|'' | None（空=未诊断）
    'human_verdict': None,  # 同上
    'verdict': None,        # 'real'|'false_alarm'|'uncertain'（综合：人/LLM 任一匹配）
    'start_ts': None,       # float | None
    'end_ts': None,         # float | None
}
_ALERT_LIMIT_DEFAULT = 20
_ALERT_LIMIT_MAX = 1000

# LLM 诊断异步线程池（请求线程不阻塞，进度走 alert_diagnose_status_api）
# 模块级单例：避免每次请求新建池
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
_diagnose_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='phm-diagnose')
# 简易进度跟踪（started/done/total/errors），与 DiagnosisService.auto_status 分离
# 这里只跟踪网页触发的批量诊断进度
_diagnose_progress: dict = {'running': False, 'done': 0, 'total': 0,
                             'errors': 0, 'started_at': 0.0, 'finished_at': 0.0}


def _parse_alert_filters(get_params):
    """从 GET 参数解析筛选条件（对齐需求书 §后台 L97 列定义）。

    返回 dict，键与 SQLiteStore.query_alerts_filtered 对齐。非法值兜底 None。

    新增 llm_verdict / human_verdict 单独过滤（需求书 §后台「LLM 诊断状态」
    「人工诊断状态」两列独立筛选）；保留 verdict 表示综合（人/LLM 任一匹配）。
    """
    out = dict(_ALERT_FILTER_DEFAULTS)

    channel = get_params.get('channel')
    if channel and isinstance(channel, str) and channel.strip():
        out['channel'] = channel.strip()[:64]  # 限长防注入

    alert_type = get_params.get('alert_type')
    if alert_type in ('measured', 'predicted', 'joint'):
        out['alert_type'] = alert_type

    # LLM 诊断筛选：支持 real/false_alarm/uncertain（已诊断三种） + 'none'（未诊断）
    llm_v = get_params.get('llm_verdict')
    if llm_v in ('real', 'false_alarm', 'uncertain', 'none'):
        out['llm_verdict'] = llm_v

    # 人工诊断筛选：同上
    human_v = get_params.get('human_verdict')
    if human_v in ('real', 'false_alarm', 'uncertain', 'none'):
        out['human_verdict'] = human_v

    # 综合状态（保留 verdict 字段：人/LLM 任一匹配）
    verdict = get_params.get('verdict')
    if verdict in ('real', 'false_alarm', 'uncertain'):
        out['verdict'] = verdict

    start_str = get_params.get('start_ts')
    if start_str:
        out['start_ts'] = _parse_iso_or_float(start_str)
    end_str = get_params.get('end_ts')
    if end_str:
        out['end_ts'] = _parse_iso_or_float(end_str)

    return out


def _parse_iso_or_float(raw):
    """把 ISO 8601 字符串或数字串解析成 Unix 时间戳（float）。

    支持 'YYYY-MM-DD' / 'YYYY-MM-DDTHH:MM:SS' / 'YYYY-MM-DD HH:MM:SS' / 1234567890.0。
    失败返回 None。
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    # 先尝试纯数字（Unix 时间戳）
    try:
        return float(s)
    except ValueError:
        pass
    # ISO 8601（含空格分隔的也支持）
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return _dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def _parse_alert_limit(raw):
    """解析 limit 参数，限幅 [1, 1000]，默认 50。"""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _ALERT_LIMIT_DEFAULT
    return max(1, min(n, _ALERT_LIMIT_MAX))


_ALERT_PAGE_DEFAULT = 1
_ALERT_PAGE_MAX = 100000  # 上限防滥用（10 万页 × 50 条/页 = 500 万条，远超实际）

# 分页栏「每页显示数量」下拉的候选值（结构化：单一真相源，前端不硬编码）
_ALERT_PAGE_SIZE_OPTIONS = [20, 50, 100, 200]


def _parse_alert_page(raw):
    """解析 page 参数，限幅 [1, _ALERT_PAGE_MAX]，默认 1。"""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _ALERT_PAGE_DEFAULT
    return max(1, min(n, _ALERT_PAGE_MAX))


def _build_page_range(page: int, total_pages: int, *, window: int = 2) -> list:
    """构建分页页码列表（当前页前后各 window 页 + 首尾 + 省略号）。

    返回 list[int | str]，其中 '..' 表示省略号。例如 page=5, total=10：
    [1, '..', 3, 4, 5, 6, 7, '..', 10]
    """
    if total_pages <= 0:
        return []
    if total_pages <= 7:
        return list(range(1, total_pages + 1))
    pages: list = []
    left = max(1, page - window)
    right = min(total_pages, page + window)
    pages.append(1)
    if left > 2:
        pages.append('..')
    for p in range(max(2, left), right + 1):
        pages.append(p)
    if right < total_pages - 1:
        pages.append('..')
    if total_pages > 1:
        pages.append(total_pages)
    # 去重（page=1 时 left=1 可能与首部重复）
    seen = set()
    deduped = []
    for p in pages:
        key = (p, len(deduped) and deduped[-1] == p)
        if p not in seen or p == '..':
            if p == '..' and deduped and deduped[-1] == '..':
                continue
            deduped.append(p)
            if p != '..':
                seen.add(p)
    return deduped


@staff_member_required
def alert_view(request):
    """告警和预警管理页（GET，measured 告警列表）。

    GET 参数（全部可选）：channel / alert_type / status / verdict /
    start_ts / end_ts / limit / page。时间戳支持 ISO 8601 或 Unix 秒。
    Container 未就绪时渲染占位页。
    """
    filters = _parse_alert_filters(request.GET)
    limit = _parse_alert_limit(request.GET.get('limit'))
    page = _parse_alert_page(request.GET.get('page'))

    c, err = _container_or_error(request)
    if c is None:
        return _render_state_page(request, err, '告警和预警管理')

    # 总行数（分页用，与 query_alerts_filtered 共用同一套筛选条件）
    try:
        total_count = c.sqlite.count_alerts_filtered(
            channel=filters['channel'],
            alert_type=filters['alert_type'],
            llm_verdict=filters['llm_verdict'],
            human_verdict=filters['human_verdict'],
            verdict=filters['verdict'],
            start_ts=filters['start_ts'],
            end_ts=filters['end_ts'],
        )
    except Exception as e:
        logger.warning("alert count_alerts_filtered failed: %s", e, exc_info=True)
        total_count = 0

    # 计算分页：page 超出 total_pages 时兜底到最后一页
    total_pages = max(1, math.ceil(total_count / limit)) if total_count > 0 else 1
    if page > total_pages:
        page = total_pages
    offset = (page - 1) * limit

    # 拉取告警（query_alerts_filtered 已按 created_at DESC 排序）
    try:
        rows = c.sqlite.query_alerts_filtered(
            channel=filters['channel'],
            alert_type=filters['alert_type'],
            llm_verdict=filters['llm_verdict'],
            human_verdict=filters['human_verdict'],
            verdict=filters['verdict'],
            start_ts=filters['start_ts'],
            end_ts=filters['end_ts'],
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.warning("alert query_alerts_filtered failed: %s", e, exc_info=True)
        rows = []

    # 传感器元信息（名+单位）
    sensor_meta = _build_sensor_meta(getattr(c, 'config', None))

    # 给每行加徽章 + 传感器名 + 单位
    decorated = []
    for r in rows:
        item = dict(r)
        item['alert_type_badge'] = _alert_type_badge(r.get('alert_type'))
        item['llm_verdict_badge'] = _verdict_badge(r.get('llm_verdict'))
        item['human_verdict_badge'] = _verdict_badge(r.get('human_verdict'))
        item['final_status_badge'] = _final_status_badge(r.get('final_status'))
        # 中文 label（与筛选栏一致，避免数据行显示英文原始值）
        item['alert_type_label'] = _label(_ALERT_TYPE_LABEL, r.get('alert_type'))
        item['llm_verdict_label'] = _label(_VERDICT_LABEL, r.get('llm_verdict')) if r.get('llm_verdict') else '未诊断'
        item['human_verdict_label'] = _label(_VERDICT_LABEL, r.get('human_verdict')) if r.get('human_verdict') else '未标注'
        item['final_status_label'] = _label(_STATUS_LABEL, r.get('final_status'))
        # raw_snapshot 末点 = 遥测值
        raw_value = None
        snap = r.get('raw_snapshot')
        if isinstance(snap, list) and snap:
            last = snap[-1]
            if isinstance(last, (int, float)):
                raw_value = float(last)
        item['raw_value'] = raw_value
        # 传感器名 + 单位
        meta = sensor_meta.get(r.get('channel'))
        item['sensor_name'] = meta['sensor_name'] if meta else (r.get('channel') or '—')
        item['unit'] = meta['unit'] if meta else ''
        decorated.append(item)

    is_superuser = request.user.is_authenticated and request.user.is_superuser

    # 当前筛选状态回填到模板（form 控件显示当前值）
    current_filters = {
        'channel': filters['channel'] or '',
        'alert_type': filters['alert_type'] or '',
        'llm_verdict': filters['llm_verdict'] or '',
        'human_verdict': filters['human_verdict'] or '',
        'verdict': filters['verdict'] or '',
        'start_ts': request.GET.get('start_ts', ''),
        'end_ts': request.GET.get('end_ts', ''),
    }

    return render(request, 'phm_site/admin/alert.html', {
        'page_title': '告警和预警管理',
        'rows': decorated,
        'row_count': len(decorated),
        'limit': limit,
        'current_filters': current_filters,
        'is_superuser': is_superuser,
        # 分页
        'total_count': total_count,
        'page': page,
        'total_pages': total_pages,
        'page_range': _build_page_range(page, total_pages),
        # 每页数量候选（供分页栏下拉渲染）
        'page_size_options': _ALERT_PAGE_SIZE_OPTIONS,
        # AJAX 端点
        'api_detail_url': reverse('phm_admin_alert_detail', args=[0]).replace('/0/', '/__ID__/'),
        'api_annotate_url': reverse('phm_admin_alert_annotate'),
        'api_delete_url': reverse('phm_admin_alert_delete'),
        'api_diagnose_url': reverse('phm_admin_alert_diagnose'),
        'api_diagnose_status_url': reverse('phm_admin_alert_diagnose_status'),
        'api_diagnose_one_url': reverse('phm_admin_alert_diagnose_one', args=[0]).replace('/0/', '/__ID__/'),
        'api_export_url': reverse('phm_admin_alert_export'),
        'api_create_url': reverse('phm_admin_alert_create'),
    })


@staff_member_required
@require_http_methods(['GET'])
def alert_detail_api(request, alert_id):
    """告警详情（抽屉用）。

    返回单条告警的完整数据：raw_snapshot / score_snapshot / 描述 / LLM
    诊断文本（从 DiagnosisService 缓存或即时调用获取）/ 当前 verdict。
    """
    try:
        aid = int(alert_id)
    except (TypeError, ValueError):
        return JsonResponse({'status': 'error', 'message': '非法 alert_id'}, status=400)
    if aid <= 0:
        return JsonResponse({'status': 'error', 'message': 'alert_id 必须 > 0'}, status=400)

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)

    # 直接查 SQLiteStore.query_alerts_filtered，限定 id（白名单内已含此参数）
    # 但 query_alerts_filtered 没有 id 参数，这里用更直接的 SQL
    try:
        row = c.sqlite.get_alert_by_id(aid)
    except AttributeError:
        # 老版本没有 get_alert_by_id，fallback 到 query_alerts_filtered 全量过滤
        all_rows = c.sqlite.query_alerts_filtered(limit=1000)
        row = next((r for r in all_rows if r.get('id') == aid), None)
    except Exception as e:
        logger.warning("alert_detail query failed: %s", e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    if not row:
        return JsonResponse({'status': 'error', 'message': f'告警 {aid} 不存在'},
                            status=404)

    # LLM 诊断文本：从 diagnosis_records 查缓存
    diagnosis_text = ''
    diagnosis_error = None
    try:
        diag = c.sqlite.get_diagnosis(row.get('channel'), row.get('alert_type'),
                                       row.get('created_at'))
        if diag:
            diagnosis_text = diag.get('diagnosis') or ''
            diagnosis_error = diag.get('error')
    except Exception as e:
        logger.debug("get_diagnosis failed: %s", e)

    # 传感器元信息（名+单位）
    sensor_meta = _build_sensor_meta(getattr(c, 'config', None))
    meta = sensor_meta.get(row.get('channel'))

    return JsonResponse({
        'status': 'ok',
        'alert': {
            'id': row.get('id'),
            'channel': row.get('channel'),
            'alert_type': row.get('alert_type'),
            'score': row.get('score'),
            'message': row.get('message') or '',
            'created_at': row.get('created_at'),
            'status': row.get('status'),
            'llm_verdict': row.get('llm_verdict'),
            'human_verdict': row.get('human_verdict'),
            'final_status': row.get('final_status'),
            'raw_snapshot': row.get('raw_snapshot'),
            'score_snapshot': row.get('score_snapshot'),
            'sensor_name': meta['sensor_name'] if meta else (row.get('channel') or '—'),
            'unit': meta['unit'] if meta else '',
        },
        'diagnosis': {
            'text': diagnosis_text,
            'error': diagnosis_error,
        },
    })


@staff_member_required
@require_http_methods(['POST'])
def alert_annotate_api(request):
    """批量标注（实警/虚警/待定）。仅超管。

    入参 JSON: {ids: [int], verdict: 'real'|'false_alarm'|'uncertain'}
    """
    ok, err_resp = _require_superuser(request)
    if not ok:
        return err_resp
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '请求体不是合法 JSON'},
                            status=400)
    ids = _parse_id_list(body.get('ids'))
    verdict = body.get('verdict')
    if not ids:
        return JsonResponse({'status': 'error', 'message': '未提供有效 id'},
                            status=400)
    if verdict not in ('real', 'false_alarm', 'uncertain'):
        return JsonResponse({'status': 'error', 'message': 'verdict 非法'},
                            status=400)

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)
    try:
        n = c.sqlite.update_alert_verdict_by_ids(ids, verdict, is_llm=False)
    except Exception as e:
        logger.warning("alert_annotate failed: %s", e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'ok', 'updated': n, 'verdict': verdict})


@staff_member_required
@require_http_methods(['POST'])
def alert_delete_api(request):
    """批量软删（移到回收站）。仅超管。

    入参 JSON: {ids: [int]}
    """
    ok, err_resp = _require_superuser(request)
    if not ok:
        return err_resp
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '请求体不是合法 JSON'},
                            status=400)
    ids = _parse_id_list(body.get('ids'))
    if not ids:
        return JsonResponse({'status': 'error', 'message': '未提供有效 id'},
                            status=400)

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)
    try:
        n = c.sqlite.delete_by_ids('alert_records', ids)
    except Exception as e:
        logger.warning("alert_delete failed: %s", e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'ok', 'deleted': n})


@staff_member_required
@require_http_methods(['POST'])
def alert_diagnose_api(request):
    """触发 LLM 诊断（staff 可用，只读语义）。单条或批量。

    入参 JSON: {ids: [int], force_refresh?: bool}
    行为：在线程池里循环调 DiagnosisService.diagnose(...)，进度走 status 端点。
    """
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '请求体不是合法 JSON'},
                            status=400)
    ids = _parse_id_list(body.get('ids'))
    force_refresh = bool(body.get('force_refresh', False))
    if not ids:
        return JsonResponse({'status': 'error', 'message': '未提供有效 id'},
                            status=400)

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)

    # 已有任务在跑：拒绝（避免并发触发多个）
    if _diagnose_progress.get('running'):
        return JsonResponse({'status': 'error',
                             'message': '已有诊断任务在跑，请等结束后再触发'},
                            status=409)

    # 取目标 alert 的 (channel, alert_type, alert_ts) 三元组
    targets = []
    for aid in ids:
        try:
            row = c.sqlite.get_alert_by_id(aid)
        except AttributeError:
            all_rows = c.sqlite.query_alerts_filtered(limit=1000)
            row = next((r for r in all_rows if r.get('id') == aid), None)
        except Exception:
            row = None
        if row and row.get('channel') and row.get('created_at'):
            targets.append((row['channel'], row.get('alert_type', 'measured'),
                            row['created_at']))

    if not targets:
        return JsonResponse({'status': 'error', 'message': '没有可诊断的告警'},
                            status=400)

    # 初始化进度并提交到线程池
    _diagnose_progress.update(
        running=True, done=0, total=len(targets), errors=0,
        started_at=_time.time(), finished_at=0.0,
    )
    _diagnose_executor.submit(_run_diagnose_batch, c, targets, force_refresh)

    return JsonResponse({'status': 'ok', 'started': True, 'total': len(targets)})


def _run_diagnose_batch(container, targets, force_refresh):
    """批量诊断 worker（线程池里执行）。

    直接调 DiagnosisService.diagnose(...) —— 它内部已带缓存（重复点不会重调
    LLM，除非 force_refresh=True）。
    """
    diag = getattr(container, 'diagnosis', None)
    if diag is None:
        _diagnose_progress.update(running=False, errors=len(targets),
                                   finished_at=_time.time())
        return
    for channel, alert_type, alert_ts in targets:
        try:
            diag.diagnose(channel, alert_type=alert_type,
                          alert_ts=alert_ts, force_refresh=force_refresh)
            _diagnose_progress['done'] += 1
        except Exception as e:
            logger.warning("diagnose failed for %s: %s", channel, e)
            _diagnose_progress['errors'] += 1
    _diagnose_progress['running'] = False
    _diagnose_progress['finished_at'] = _time.time()


@staff_member_required
@require_http_methods(['GET'])
def alert_diagnose_status_api(request):
    """查询诊断进度。staff 可用。"""
    return JsonResponse({'status': 'ok', 'progress': dict(_diagnose_progress)})


@staff_member_required
@require_http_methods(['POST'])
def alert_diagnose_one_api(request, alert_id):
    """单条同步诊断（抽屉内「诊断/重新诊断」按钮用）。

    与批量异步的 alert_diagnose_api 不同，这里直接调
    DiagnosisService.diagnose()（同步返回），适合单条即时反馈场景。
    DiagnosisService 内部已带缓存（force_refresh=False 时命中缓存秒回），
    并自动把 llm_verdict 写回 alert_records。

    入参 JSON: {force_refresh?: bool}
    返回: {status, diagnosis_text, llm_verdict, error, elapsed_sec, cached}
    """
    try:
        aid = int(alert_id)
    except (TypeError, ValueError):
        return JsonResponse({'status': 'error', 'message': '非法 alert_id'}, status=400)
    if aid <= 0:
        return JsonResponse({'status': 'error', 'message': 'alert_id 必须 > 0'}, status=400)

    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        body = {}
    force_refresh = bool(body.get('force_refresh', True))  # 单条默认强制刷新（用户主动点）

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)

    # 取告警三元组
    try:
        row = c.sqlite.get_alert_by_id(aid)
    except AttributeError:
        all_rows = c.sqlite.query_alerts_filtered(limit=1000)
        row = next((r for r in all_rows if r.get('id') == aid), None)
    except Exception as e:
        logger.warning("alert_diagnose_one query failed: %s", e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    if not row:
        return JsonResponse({'status': 'error', 'message': f'告警 {aid} 不存在'}, status=404)
    if not row.get('channel') or not row.get('created_at'):
        return JsonResponse({'status': 'error', 'message': '告警缺少 channel/created_at，无法诊断'},
                            status=400)

    diag = getattr(c, 'diagnosis', None)
    if diag is None:
        return JsonResponse({'status': 'error', 'message': '诊断服务未初始化'}, status=503)

    try:
        result = diag.diagnose(
            row['channel'],
            alert_type=row.get('alert_type', 'measured'),
            alert_ts=row['created_at'],
            force_refresh=force_refresh,
        )
    except Exception as e:
        logger.warning("alert_diagnose_one failed for %s: %s", aid, e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    # 重新读最新行，拿到 DiagnosisService 写回的 llm_verdict / final_status
    try:
        fresh = c.sqlite.get_alert_by_id(aid) or {}
    except Exception:
        fresh = {}

    return JsonResponse({
        'status': 'ok',
        'id': aid,
        'diagnosis_text': result.get('diagnosis') or '',
        'llm_verdict': fresh.get('llm_verdict') or result.get('llm_verdict'),
        'final_status': fresh.get('final_status'),
        'error': result.get('error'),
        'elapsed_sec': result.get('elapsed_sec', 0),
        'cached': result.get('cached', False),
    })


@staff_member_required
@require_http_methods(['GET'])
def alert_export_api(request):
    """导出告警为 CSV/JSON。

    GET 参数：
      - format: 'csv'（默认，UTF-8 BOM）/ 'json'
      - ids: 逗号分隔（可选，指定导出某些 id；不填则按当前筛选）
      - 其他筛选参数同 alert_view（channel/alert_type/status/verdict/时间窗）
    """
    fmt = request.GET.get('format', 'csv').lower()
    if fmt not in ('csv', 'json'):
        fmt = 'csv'

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)

    # 优先按 ids 导出；否则按筛选参数（与列表页一致）
    ids_raw = request.GET.get('ids')
    ids = _parse_id_list(ids_raw) if ids_raw else []
    if ids:
        # 按 id 列表查（无筛选）
        rows = []
        for aid in ids:
            try:
                row = c.sqlite.get_alert_by_id(aid)
            except AttributeError:
                continue
            except Exception:
                continue
            if row:
                rows.append(row)
    else:
        filters = _parse_alert_filters(request.GET)
        # 导出放宽 limit 上限到 10 万（需求书 L114：每通道上限 10 万行）
        rows = c.sqlite.query_alerts_filtered(
            channel=filters['channel'],
            alert_type=filters['alert_type'],
            llm_verdict=filters['llm_verdict'],
            human_verdict=filters['human_verdict'],
            verdict=filters['verdict'],
            start_ts=filters['start_ts'],
            end_ts=filters['end_ts'],
            limit=100000,
        )

    # 序列化为 5 列（需求书 L114）：channel/timestamp/raw_value/anomaly_score/received_at_iso
    def _iso(ts):
        if ts is None:
            return ''
        try:
            return _dt.datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            return ''

    def _raw_value(snap):
        if isinstance(snap, list) and snap:
            last = snap[-1]
            if isinstance(last, (int, float)):
                return float(last)
        return ''

    serialised = [{
        'channel': r.get('channel') or '',
        'timestamp': r.get('created_at') or '',
        'raw_value': _raw_value(r.get('raw_snapshot')),
        'anomaly_score': r.get('score') if r.get('score') is not None else '',
        'received_at_iso': _iso(r.get('ingested_at') or r.get('created_at')),
    } for r in rows]

    if fmt == 'json':
        payload = json.dumps({'count': len(serialised), 'alerts': serialised},
                             ensure_ascii=False, indent=2)
        resp = HttpResponse(payload, content_type='application/json; charset=utf-8')
        resp['Content-Disposition'] = (
            'attachment; filename="phm_alerts.json"'
        )
        return resp

    # CSV（UTF-8 BOM，Excel 直开）
    import csv
    import io as _io

    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['channel', 'timestamp', 'raw_value', 'anomaly_score',
                     'received_at_iso'])
    for row in serialised:
        writer.writerow([row['channel'], row['timestamp'], row['raw_value'],
                         row['anomaly_score'], row['received_at_iso']])

    # 流式响应（大文件友好）
    def _stream():
        yield b'\xef\xbb\xbf'  # UTF-8 BOM
        yield buf.getvalue().encode('utf-8')

    resp = StreamingHttpResponse(_stream(), content_type='text/csv; charset=utf-8')
    resp['Content-Disposition'] = 'attachment; filename="phm_alerts.csv"'
    return resp


@staff_member_required
@require_http_methods(['POST'])
def alert_create_api(request):
    """人工补录告警（漏检时用）。仅超管。

    入参 JSON: {channel: str, score: float, message?: str,
                created_at?: float (ISO 字符串或 Unix 秒),
                raw_snapshot?: list[float], score_snapshot?: list[float]}
    """
    ok, err_resp = _require_superuser(request)
    if not ok:
        return err_resp
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '请求体不是合法 JSON'},
                            status=400)

    channel = body.get('channel')
    if not channel or not isinstance(channel, str):
        return JsonResponse({'status': 'error', 'message': 'channel 必填且为字符串'},
                            status=400)

    try:
        score = float(body.get('score', 0))
    except (TypeError, ValueError):
        return JsonResponse({'status': 'error', 'message': 'score 必须为数字'},
                            status=400)

    message = str(body.get('message') or '')
    created_at = _parse_iso_or_float(body.get('created_at'))
    raw_snapshot = body.get('raw_snapshot')
    if raw_snapshot is not None and not isinstance(raw_snapshot, list):
        return JsonResponse({'status': 'error', 'message': 'raw_snapshot 必须为数组'},
                            status=400)
    score_snapshot = body.get('score_snapshot')
    if score_snapshot is not None and not isinstance(score_snapshot, list):
        return JsonResponse({'status': 'error', 'message': 'score_snapshot 必须为数组'},
                            status=400)

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)

    try:
        new_id = c.sqlite.insert_alert_manual(
            channel=channel, score=score, message=message,
            created_at=created_at,
            raw_snapshot=raw_snapshot,
            score_snapshot=score_snapshot,
        )
    except Exception as e:
        logger.warning("alert_create failed: %s", e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    if new_id is None:
        return JsonResponse({'status': 'error', 'message': '插入失败'}, status=500)
    return JsonResponse({'status': 'ok', 'id': new_id})


# ════════════════════════════════════════════════════════════════════════════
# 第 6 页：设备树管理（左树 + 右编辑面板 + 拖拽 3 语义 + 防成环）
# ════════════════════════════════════════════════════════════════════════════
# 需求书 §后台「设备树管理」：
#   左树（新建文件夹/传感器/拖拽 3 种语义 + 防成环）+ 右编辑面板
#   （名称/健康度计算方式/数据源下拉/传输块大小/描述/@命令/应用/删除）
#   + 特殊传感器 * 标注 + 不参与轮播。
#
# 决策（已确认）：
#   - 模型绑定延续 @ 命令隐式（schema 不改）
#   - service 层零改动：ConfigService.save 已就绪（空树保护 + 重复 sourceId 校验 + TCP 推送）
#   - 拖拽防成环在前端 JS 实现（DFS 检测祖先链），后端不重做

# space_daq_channels.json 位置（与 system_config 同目录）
_SPACE_CHANNELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "space_daq_channels.json",
)


def _load_space_channels():
    """读取 space_daq_channels.json，给设备树「数据源下拉」用。

    返回 [{id, source_id, label, enabled}] 列表。文件缺失返回 []。
    """
    if not os.path.exists(_SPACE_CHANNELS_PATH):
        return []
    try:
        with open(_SPACE_CHANNELS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        channels = raw.get('channels', []) if isinstance(raw, dict) else []
        out = []
        for ch in channels:
            if not isinstance(ch, dict):
                continue
            out.append({
                'id': ch.get('id'),
                'source_id': ch.get('source_id') or ch.get('sourceId') or '',
                'label': ch.get('label') or ch.get('source_id') or '',
                'enabled': bool(ch.get('enabled', True)),
            })
        return out
    except Exception:
        logger.warning("failed to read space_daq_channels.json", exc_info=True)
        return []


def _mark_special_sensors(nodes):
    """递归给 sensor 加 _special 标记（含 @rul 命令或 isSpecial=true）。

    返回新的树（深拷贝，避免污染 service 内存）。前端依据此标记加 * 标注。
    """
    if not isinstance(nodes, list):
        return []
    import copy
    out = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        m = copy.deepcopy(n)
        if m.get('type') == 'sensor':
            desc = m.get('description') or ''
            is_special = bool(m.get('isSpecial')) or '@rul' in desc
            m['_special'] = is_special
        children = m.get('children')
        if children:
            m['children'] = _mark_special_sensors(children)
        out.append(m)
    return out


@staff_member_required
def device_tree_view(request):
    """设备树管理页（GET）。

    Container 未就绪时渲染占位页。Container 就绪后渲染左树 + 右空面板，
    前端 JS 负责选中节点、编辑、拖拽、防成环、保存。
    """
    c, err = _container_or_error(request)
    if c is None:
        return _render_state_page(request, err, '设备树管理')

    try:
        cfg = c.config.load()
        tree = _mark_special_sensors(cfg.get('device_tree') or [])
    except Exception as e:
        logger.warning("device_tree load failed: %s", e, exc_info=True)
        tree = []

    channels = _load_space_channels()

    is_superuser = request.user.is_authenticated and request.user.is_superuser
    aggregation_strategy = cfg.get('aggregation_strategy', 'min') if isinstance(cfg, dict) else 'min'

    return render(request, 'phm_site/admin/device_tree.html', {
        'page_title': '设备树管理',
        'tree_json': json.dumps(tree, ensure_ascii=False),
        'channels_json': json.dumps(channels, ensure_ascii=False),
        'aggregation_strategy': aggregation_strategy,
        'is_superuser': is_superuser,
        'save_url': reverse('phm_admin_device_tree_save'),
        'channels_api_url': reverse('phm_admin_device_tree_channels'),
    })


@staff_member_required
@require_http_methods(['GET'])
def device_tree_space_channels_api(request):
    """数据源下拉的数据源（GET）。

    返回 space_daq_channels.json 内容（与 device_tree_view 注入的 channels_json 同源）。
    提供独立端点供前端动态刷新（管理员改了 space_daq_channels.json 后无需 reload 页面）。
    """
    return JsonResponse({
        'status': 'ok',
        'channels': _load_space_channels(),
    })


@staff_member_required
@require_http_methods(['POST'])
def device_tree_save_api(request):
    """保存整树（POST，仅超管）。

    入参 JSON: 整个 device_config.json 的 body（含 device_tree + aggregation_strategy）。
    也可以只传 {device_tree: [...]}（aggregation_strategy 缺省时保留旧值）。
    """
    ok, err_resp = _require_superuser(request)
    if not ok:
        return err_resp
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '请求体不是合法 JSON'},
                            status=400)
    if not isinstance(body, dict):
        return JsonResponse({'status': 'error', 'message': '请求体必须是 JSON 对象'},
                            status=400)
    if 'device_tree' not in body:
        return JsonResponse({'status': 'error', 'message': '缺少 device_tree 字段'},
                            status=400)
    tree = body.get('device_tree')
    if not isinstance(tree, list):
        return JsonResponse({'status': 'error', 'message': 'device_tree 必须为数组'},
                            status=400)

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)

    try:
        result = c.config.save(body)
    except Exception as e:
        logger.warning("device_tree save failed: %s", e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    if result.get('status') != 'ok':
        return JsonResponse(result, status=400)
    return JsonResponse(result)
