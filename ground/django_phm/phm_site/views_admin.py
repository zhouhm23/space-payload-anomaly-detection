"""Admin custom-page views (/admin/phm_site/<page>/).

Spec (admin section) — 9 pages total:
  - login / home / user management / audit log: SimpleUI defaults, not in this file
  - dashboard / alert-management / recycle / device-tree / system-settings /
    model-management: implemented here

Design notes:
  - Every page view is gated by ``@staff_member_required`` (spec: "show login page when not logged in").
  - Reuses the ``views_api._container_or_503`` three-state idea but returns a Django HttpResponse.
  - AJAX actions (annotate / delete / save) live as JSON views in the same file,
    with paths like ``/admin/phm_site/<page>/api/<action>/``.
  - All business logic goes through the Service layer (ConfigService /
    SQLiteStore / ...); nothing is reimplemented in the views.
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


# ── Common helpers ──────────────────────────────────────────────────────────
def _container_or_error(request):
    """Get the Container; return (None, error_context_dict) when not ready.

    Unlike the API, admin pages do not return 503 — they render a friendly
    "initialising" placeholder so the admin sees the system state instead of
    waiting blind. A flag is still returned for the caller to check.
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
    """Render the placeholder page shown when the Container is not ready."""
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
    """Superuser check. Returns (ok, error_response)."""
    if not request.user.is_authenticated or not request.user.is_superuser:
        return False, JsonResponse(
            {'status': 'error', 'message': '仅管理员可执行此操作'},
            status=403,
        )
    return True, None


# ════════════════════════════════════════════════════════════════════════════
# Page 1: Model management (read-only cards)
# ════════════════════════════════════════════════════════════════════════════
# Spec (admin section "Model management"):
#   Anomaly-detection / forecasting / other dedicated model cards. Show their
#   info and which sensors use them. They cannot be added or deleted from the
#   web UI — that needs a config-file change. No enable / disable / reload.
#
# Data sources:
#   - phm.algorithm._registry.MODEL_REGISTRY (pure metadata, no torch import)
#   - @ commands in the device-tree description (e.g. @异常检测模型 / @预测模型 / @rul:xxx)
#   - Local asset existence check (HF cache snapshot / RUL weight files)

# Model key → Chinese role name
_KIND_LABEL = {
    'detector': '异常检测',
    'forecaster': '趋势预测',
    'rul': '退化预测(RUL)',
}
# Deployment label (space segment reserves OTA; ground segment runs local inference)
_DEPLOY_LABEL = {
    'ground': '地基',
    'space': '天基',
}

# @ command → model key mapping (spec addendum: a sensor may @异常检测模型 / @预测模型 / @专属模型).
# Supported @ command prefixes → registry key. Matched by prefix when scanning descriptions.
_AT_COMMAND_MAP = {
    '@tspulse': 'tspulse',
    '@异常检测模型': 'tspulse',
    '@ttm': 'ttm_r3',
    '@预测模型': 'ttm_r3',
    '@rul': 'rul',
}


def _scan_sensor_model_usage(device_tree):
    """Scan @ commands in the device-tree descriptions; return {model_key: [sensor_name, ...]}."""
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
    """Check whether a model's local assets exist (no torch import).

    Returns {'available': bool, 'path': str, 'note': str}.
    The HF cache path is resolved by ``_hf_cache.resolve_local_model_path``
    (which reads the ``HF_HOME`` env var that settings.py has already
    ``setdefault``-ed); here we only do an existence check.
    """
    entry = get_model_entry(model_key)
    if entry is None:
        return {'available': False, 'path': '', 'note': '未知模型'}

    if entry.hub_id:
        # HF model: check whether a snapshot dir exists under .hf_cache.
        # resolve_local_model_path handles path resolution; here we only check existence.
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

    # Local-weight model (RUL): check for weight files under models/rul/
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
    """Model-management page (read-only cards).

    Shows each model's metadata from MODEL_REGISTRY + local asset status +
    which sensors reference it.
    """
    # Device-tree usage scan (runs even when the Container is not ready — reads JSON directly)
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

    # Default usage: sensors without an explicit @ command use the system default (detector→tspulse, forecaster→ttm_r3)
    default_usage = _scan_default_usage(device_tree)

    cards = []
    for key, entry in MODEL_REGISTRY.items():
        assets = _check_local_assets(key)
        # Merge explicit @ usage + default usage
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
    """Sensors without an explicit @ command: normal sensors default to
    tspulse + ttm_r3, special sensors (isSpecial) default to rul.

    Returns ``{model_key: [sensor_name, ...]}``.
    """
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
                    # Normal sensor: only counted as default when it has no explicit @ command
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
# Page 2: Dashboard (health banner + three cards + alert-trend bar chart + time-window switch)
# ════════════════════════════════════════════════════════════════════════════
# Spec (admin section "Dashboard"):
#   A header banner shows overall system health; the middle shows three cards —
#   human-diagnosed alerts (incl. warnings), LLM-diagnosed alerts, and
#   undiagnosed alert counts; below is an alert-trend bar chart (bucketed by
#   unit time). The three cards and the bar chart can all switch between today,
#   last 7 days, last 30 days.
#
# Data sources:
#   - Container.health.system_health() → banner health ([0,1], ×100 when shown)
#   - AlertRecord ORM (db_table=alert_records, same table as SQLiteStore)
#     time-window filter + Python-side three-way classification + bucket aggregation
#
# Design notes:
#   - Time-window switching uses SSR GET params (?window=today|7d|30d), no AJAX —
#     the spec explicitly says "the UI does not update live; refresh the page",
#     consistent with the alert-management page.
#   - The bar chart uses pure CSS bars, no ECharts (the front-end monitor already
#     has it; the admin stays lightweight).
#   - All aggregation is done in Python; the Service layer is untouched
#     (SQLiteStore is stable).
#   - Health-tier thresholds are hardcoded for now (can be externalised to
#     system_config.json later if needed).

# Time-window choices (legal values for the GET param `window`)
_WINDOW_CHOICES = ('today', '7d', '30d')
_WINDOW_DEFAULT = 'today'

# Time-window tab config (key → Chinese label)
_WINDOW_TABS = (
    ('today', '今天'),
    ('7d', '最近 7 天'),
    ('30d', '最近 30 天'),
)

# Health tiers (threshold, tier_key, tier_text) — corresponds to admin.css's
# .phm-dash-banner-{tier} colours. Thresholds are hardcoded for now; to make
# them tunable online later, externalise them to system_config.json's dashboard
# section (see the SystemConfigService pattern).
_HEALTH_TIERS = (
    (0.80, 'normal',  '系统正常'),
    (0.50, 'warning', '存在告警'),
    (0.00, 'danger',  '健康度低'),
)

# Auto-refresh interval (seconds). The dashboard auto-refreshes by default
# (enabled even without the `auto` URL param); the user can turn it off via the
# checkbox (writes ?auto=0). The front-end uses setInterval to count down and reload.
_DASHBOARD_REFRESH_SECONDS = 15


def _health_tier(system_value):
    """Map a [0,1] health value to a banner state tier.

    Returns (tier_key, tier_text). tier_key is used as a CSS class name
    (normal/warning/danger).
    """
    for threshold, key, text in _HEALTH_TIERS:
        if system_value >= threshold:
            return key, text
    return _HEALTH_TIERS[-1][1], _HEALTH_TIERS[-1][2]


def _window_bounds(window, now=None):
    """Compute the time window's [start_ts, end_ts] and bucket config.

    Returns (start_ts, end_ts, bucket_kind, bucket_count):
      - today: today 00:00 → now, bucketed by hour (24 buckets)
      - 7d:    today 00:00 minus 6 days → now, bucketed by day (7 buckets, incl. today)
      - 30d:   today 00:00 minus 29 days → now, bucketed by day (30 buckets, incl. today)

    An unknown window value falls through to the today branch.
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
    else:  # 'today' or unknown value fallback
        start_dt = today_start
        bucket_kind, bucket_count = 'hour', 24
    return start_dt.timestamp(), now, bucket_kind, bucket_count


def _classify_verdict(human_v, llm_v):
    """Classify alert diagnosis status into three categories.

    Returns 'human' / 'llm' / 'undiagnosed':
      - human_verdict is non-empty ('real'/'false_alarm'/'uncertain') → 'human'
      - otherwise llm_verdict is non-empty → 'llm'
      - both empty (None or '') → 'undiagnosed'

    Both empty strings and None are treated as "unlabeled" (VERDICT_CHOICES first
    item is ''). Priority matches ``AlertRecord.final_status``: human > LLM.
    """
    if human_v:
        return 'human'
    if llm_v:
        return 'llm'
    return 'undiagnosed'


def _bucket_index(ts, start_ts, bucket_kind):
    """Compute the bucket index (0-based) for timestamp *ts*.

    bucket_kind='hour' → 3600 s/bucket; 'day' → 86400 s/bucket.
    """
    span = 3600 if bucket_kind == 'hour' else 86400
    return int((ts - start_ts) // span)


def _format_bucket_label(idx, bucket_kind, start_ts):
    """Short display label for bucket *idx* (used as x-axis tick).

    - hour: bare hour number, e.g. '14' (24-bucket compact display; hover title
      still gives full info)
    - day: bare day number, e.g. '21' (7/30 buckets also compact)
    """
    span = 3600 if bucket_kind == 'hour' else 86400
    bucket_start = start_ts + idx * span
    dt = _dt.datetime.fromtimestamp(bucket_start)
    return str(dt.hour) if bucket_kind == 'hour' else str(dt.day)


def _format_bucket_title(idx, bucket_kind, start_ts):
    """Mouse-hover title for bucket *idx* (full info, no truncation).

    - hour: '2026-07-21 14:00'
    - day: '2026-07-21'
    """
    span = 3600 if bucket_kind == 'hour' else 86400
    bucket_start = start_ts + idx * span
    dt = _dt.datetime.fromtimestamp(bucket_start)
    return dt.strftime('%Y-%m-%d %H:00') if bucket_kind == 'hour' else dt.strftime('%Y-%m-%d')


def _collect_dashboard_metrics(window, alerts):
    """Aggregate dashboard statistics metrics.

    Args:
        window: Time-window key ('today', '7d', '30d').
        alerts: Iterable where each item has created_at / human_verdict /
            llm_verdict attributes (AlertRecord ORM or duck-typed object).

    Returns:
        dict with keys:
        - window, start_ts, end_ts, bucket_kind,
        - counts: {human, llm, undiagnosed, total},
        - breakdown: verdict sub-totals per source category (human / llm)
            each with {real, false_alarm, uncertain},
        - buckets: list of bucket dicts (including zero-count buckets),
          each with label, title, count, and parts (source x verdict matrix
          for frontend stacked-bar chart).
    """
    start_ts, end_ts, bucket_kind, bucket_count = _window_bounds(window)
    counts = {'human': 0, 'llm': 0, 'undiagnosed': 0, 'total': 0}
    breakdown = {
        'human': {'real': 0, 'false_alarm': 0, 'uncertain': 0},
        'llm':   {'real': 0, 'false_alarm': 0, 'uncertain': 0},
    }
    bucket_counts = [0] * bucket_count
    # Per-bucket source x verdict matrix (stacked-bar data source)
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
        # Discard timestamps outside the window (ORM already filters;
        # defensive: mock/historical calls may pass out-of-range data)
        if ts < start_ts or ts > end_ts:
            continue
        category = _classify_verdict(a.human_verdict, a.llm_verdict)
        counts[category] += 1
        counts['total'] += 1
        # Verdict sub-totals: human category uses human_verdict,
        # llm category uses llm_verdict
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
    """Dashboard page.

    GET ?window=today|7d|30d switches the time window (default today; invalid
    values fall through). When the Container is not ready, renders a
    placeholder page (_state.html) instead of returning 500.
    """
    window = request.GET.get('window', _WINDOW_DEFAULT)
    if window not in _WINDOW_CHOICES:
        window = _WINDOW_DEFAULT

    # Auto-refresh: enabled by default (?auto=0 to disable). Per requirements doc,
    # dashboard auto-refreshes by default.
    auto_refresh = request.GET.get('auto', '1') != '0'

    # Container three-state gate
    c, err = _container_or_error(request)
    if c is None:
        return _render_state_page(request, err, '仪表盘')

    # Health score ([0,1] → multiplied by 100 for display)
    try:
        health_data = c.health.system_health()
    except Exception as e:
        logger.warning("system_health failed: %s", e)
        health_data = {'system': 1.0, 'channels': {}, 'threshold': 0}

    system_value = float(health_data.get('system', 1.0))
    tier_key, tier_text = _health_tier(system_value)

    # Ground-Cloud link status (same source as the front-end top-bar
    # system_info_view: services_bridge.get_link_status)
    # link_status: {rtt_ms, status, last_success_ts}. status: online/degraded/offline/waiting
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

    # Alert time-window aggregation
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

    # Time-window tabs (active marks the currently selected tab)
    tabs = [
        {'key': k, 'label': lbl, 'active': (k == window)}
        for k, lbl in _WINDOW_TABS
    ]

    # Bar-chart max value (for CSS height ratio; template shows empty state when 0)
    # max_bucket = max single-bucket total count (frontend renders stacked bar by segment ratio)
    max_bucket = max((b['count'] for b in metrics['buckets']), default=0)

    # Serialize buckets into a structure frontend JS can consume
    # (parts matrix + label/title/count). Template SSR-outputs a copy of the
    # JSON for JS, avoiding an extra fetch from JS.
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
# Page 3: Recycle bin (only super-admin can modify)
# ════════════════════════════════════════════════════════════════════════════
# Requirements doc (admin section "Recycle bin (super-admin only)"):
#   Same list shape as the alert management page, but the action bar only has
#   "Permanent Delete" + "Restore" buttons.
#
# Data source: is_deleted=1 rows from the three SQLiteStore business tables
#   (detection_results / alert_records / diagnosis_records). Section 1 public
#   preamble already added query_deleted / restore / purge_by_ids methods;
#   this view is a thin wrapper.

# URL ?table= whitelist (key → (SQLiteStore table name, display label))
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
    """Parse the ?table= parameter. Returns (table_key, sql_table, label).

    Invalid values fall back to alerts (default tab). Returns a triple shared
    by the view and the template.
    """
    if key not in _RECYCLE_TABLE_MAP:
        key = _RECYCLE_TABLE_DEFAULT
    sql_table, label = _RECYCLE_TABLE_MAP[key]
    return key, sql_table, label


def _parse_recycle_limit(raw):
    """Parse the ?limit= parameter, clamped to [1, 1000]. Invalid values fall back to default."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _RECYCLE_LIMIT_DEFAULT
    return max(1, min(n, _RECYCLE_LIMIT_MAX))


def _parse_id_list(raw):
    """Normalize the ids from a request (list or comma-separated string) into list[int].

    Supports JSON array / comma-separated string / single id.
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
    """verdict → CSS badge class (consistent with dashboard color order:
    real=red / false_alarm=green / uncertain=yellow)."""
    return {
        'real':        'phm-badge-red',
        'false_alarm': 'phm-badge-green',
        'uncertain':   'phm-badge-yellow',
    }.get(verdict, 'phm-badge-gray')


def _alert_type_badge(alert_type):
    """alert_type → CSS badge class (measured=red / predicted=yellow / joint=purple)."""
    # Note: joint alert uses cyan to distinguish from the two-color scheme
    return {
        'measured':  'phm-badge-red',
        'predicted': 'phm-badge-yellow',
        'joint':     'phm-badge-cyan',
    }.get(alert_type, 'phm-badge-gray')


def _build_sensor_meta(config_service):
    """Build a {channelName: {sensor_name, unit}} mapping from device_tree.

    The requirements-doc admin page "Alert and warning management" needs the
    list columns to include "sensor name" (distinct from the channel name
    channelName; in device_tree this is the sensor.name field, usually the same
    as channelName but independently configurable) plus "telemetry value + unit"
    display. The recycle-bin list reuses the alert-management column
    definitions, so it also needs these two fields.
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
    """Comprehensive status final_status → CSS badge class.

    final_status priority: human > llm > verification (active/pending/confirmed/false).
    real→red / false_alarm→green / uncertain→yellow / confirmed→blue / false→green /
    pending→yellow / active→gray.
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


# ── Chinese label mapping (single source of truth, shared by alert_view + recycle_view) ──
# Solves the mixed Chinese/English problem: data rows output raw English DB values
# (measured/real/active), filter bars use Chinese, JS partial refresh also uses
# Chinese — SSR and JS are inconsistent. Here the backend uniformly provides
# *_label fields; both the template and JS use labels to stay consistent.
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
# Comprehensive status / alert status Chinese mapping (final_status + status shared)
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
    """Look up the Chinese label from *mapping*. Falls back to the raw value
    if not found (None falls back to '—')."""
    if not value:
        return '—'
    return mapping.get(value, value)


@staff_member_required
def recycle_view(request):
    """Recycle-bin page (GET).

    GET ?table=alerts|detections|diagnoses switches the three resources
    (default alerts). GET ?limit=N controls rows per page (1-1000, default 200).
    When the Container is not ready, renders a placeholder page (_state.html)
    instead of returning 500.

    Column definitions align with the requirements-doc admin page
    "Alert and warning management" (10 columns) minus the "Action" column
    (recycle bin has no drawer actions; the action bar is uniformly "Restore" +
    "Permanent Delete"), plus "Deletion Time" (recycle-bin-specific). I.e.:
    checkbox / id / type / sensor name / telemetry value / anomaly score /
    alert time / LLM status / human status / comprehensive status / deletion time.
    """
    table_key, sql_table, label = _parse_recycle_table(request.GET.get('table'))
    limit = _parse_recycle_limit(request.GET.get('limit'))
    page = _parse_alert_page(request.GET.get('page'))  # Reuse generic page parser

    c, err = _container_or_error(request)
    if c is None:
        return _render_state_page(request, err, '回收站')

    # Total row count (for pagination)
    try:
        total_count = c.sqlite.count_deleted(sql_table)
    except Exception as e:
        logger.warning("recycle count_deleted(%s) failed: %s", sql_table, e)
        total_count = 0

    # Calculate pagination: clamp page to last page if it exceeds total_pages
    total_pages = max(1, math.ceil(total_count / limit)) if total_count > 0 else 1
    if page > total_pages:
        page = total_pages
    offset = (page - 1) * limit

    # Call SQLiteStore.query_deleted (ordered by deleted_at DESC)
    try:
        rows = c.sqlite.query_deleted(sql_table, limit=limit, offset=offset)
    except Exception as e:
        logger.warning("recycle query_deleted(%s) failed: %s", sql_table, e)
        rows = []

    # Sensor metadata (used by alert list: sensor name + unit)
    sensor_meta = _build_sensor_meta(getattr(c, 'config', None))

    # Add badge classes to each row (template uses them directly)
    decorated = []
    for r in rows:
        item = dict(r)
        item['alert_type_badge'] = _alert_type_badge(r.get('alert_type'))
        item['llm_verdict_badge'] = _verdict_badge(r.get('llm_verdict'))
        item['human_verdict_badge'] = _verdict_badge(r.get('human_verdict'))
        item['final_status_badge'] = _final_status_badge(r.get('final_status'))
        # Chinese labels (consistent with filter bars; avoid showing raw English values in data rows)
        item['alert_type_label'] = _label(_ALERT_TYPE_LABEL, r.get('alert_type'))
        item['llm_verdict_label'] = _label(_VERDICT_LABEL, r.get('llm_verdict')) if r.get('llm_verdict') else '未诊断'
        item['human_verdict_label'] = _label(_VERDICT_LABEL, r.get('human_verdict')) if r.get('human_verdict') else '未标注'
        item['final_status_label'] = _label(_STATUS_LABEL, r.get('final_status'))
        # Sensor name + unit (used by alert list columns)
        meta = sensor_meta.get(r.get('channel'))
        item['sensor_name'] = meta['sensor_name'] if meta else (r.get('channel') or '—')
        item['unit'] = meta['unit'] if meta else ''
        decorated.append(item)

    # Three tabs (active marks the currently selected tab)
    tabs = [
        {'key': k, 'label': lbl[1], 'active': (k == table_key)}
        for k, lbl in _RECYCLE_TABLE_MAP.items()
    ]

    # Whether the current user is a super-admin (template shows/hides bulk action buttons based on this)
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
        # Pagination (same style as alert_view, reuses _pagination.html / _page_size_select.html)
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
    """Restore soft-deleted records (POST, super-admin only).

    Input JSON: {table: 'alerts|detections|diagnoses', ids: [int,...]}
    Output JSON: {status: 'ok', restored: N} or {status: 'error', message}
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
    """Permanently delete (physically remove) soft-deleted records (POST, super-admin only).

    Input JSON: {table: 'alerts|detections|diagnoses', ids: [int,...]}
    Output JSON: {status: 'ok', purged: N} or {status: 'error', message}

    Safety constraint: purge_by_ids only deletes is_deleted=1 rows; active data
    is not affected.
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
# Page 9: User management + Audit log (SimpleUI defaults + permissions help page)
# ════════════════════════════════════════════════════════════════════════════
# Requirements doc (admin section):
#   - "User management (SimpleUI default), add a help button that opens a
#     permissions explanation panel"
#   - "Audit log (SimpleUI default)"
#
# Implementation strategy:
#   - User management (User/Group CRUD): entirely via SimpleUI defaults
#     (django.contrib.auth already registered)
#   - Audit log (LogEntry browsing): entirely via SimpleUI defaults
#     (django.contrib.admin already registered). Scope confirmed: Django
#     LogEntry only records admin-site ModelAdmin CRUD, not custom-page AJAX
#     operations (user has confirmed this is an accepted boundary)
#   - Only addition: permissions help static page /admin/phm_site/permissions/

# Role list (aligned with _require_superuser / @staff_member_required checks)
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

# Permission matrix per feature page: {operation: {role: '✓' / 'read-only' / '—'}}
# Aligned with actual checks in _require_superuser helper + @staff_member_required
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

# Audit log scope notes (confirmed accepted boundary by the user)
_AUDIT_SCOPE_NOTES = [
    'Django LogEntry 默认仅记录 admin 站内 ModelAdmin 的增删改操作（用户/组/业务模型列表页）。',
    '自定义页的 AJAX 写操作（如回收站恢复/永久删除、告警标注、系统设置保存、设备树保存）当前<strong>不</strong>写入 LogEntry。',
    'CLI 命令（manage.py phm_*）与 API 调用（/api/v2/*）也<strong>不</strong>计入 LogEntry。',
    '如需扩展到自定义页操作，需在各自定义页 view 里手动 log_action()（未来工作）。',
]


@staff_member_required
def permissions_view(request):
    """Permissions help static page (GET).

    Pure SSR, no Container dependency. Displays the permission matrix for three
    roles (anonymous / staff / superuser), helping admins quickly understand
    what each role can do and what permissions each page requires.
    """
    # Current user role (highlight the current row)
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
# Page 5: System settings (system config / front-end theme / channel calibration)
# ════════════════════════════════════════════════════════════════════════════
# Requirements doc (admin section "System settings (super-admin only)"):
#   Contains system config, front-end theme, and channel calibration categories.
#   Various config items with Chinese display names (hover descriptions),
#   variable names, and values.
#
# Data sources:
#   - system: SystemConfigService.raw_with_docs() / save()
#   - theme: ThemeService.raw_with_docs() / save()
#   - calibration: directly reads channel_calibration.json (read-only, offline
#     calibration product)
#
# Design notes:
#   - Channel calibration is read-only: Day15 offline LOO calibration product;
#     online editing would corrupt the calibration baseline
#   - Keys managed via .env (e.g. llm.timeout_sec) are shown as disabled in the UI
#   - Save takes effect via atomic write + load() hot-reload, no restart needed

# Three category tabs (key → (label, service_kind))
_SETTINGS_CATEGORIES = (
    ('system',      '系统配置'),
    ('theme',       '前台主题'),
    ('calibration', '通道校准（只读）'),
)
_SETTINGS_CATEGORY_DEFAULT = 'system'
_SETTINGS_CATEGORY_KEYS = frozenset(k for k, _ in _SETTINGS_CATEGORIES)

# Channel calibration file location (same source as CalibrationConfig.DEFAULT_CONFIG_PATH,
# but reads the raw file directly).
# __file__ = src/ground/django_phm/phm_site/views_admin.py
# Up 3 levels → src/ground/, then join data/channel_calibration.json
# (same paradigm as SystemConfigService).
_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "channel_calibration.json",
)


def _parse_settings_category(raw):
    """Parse the ?category= parameter. Invalid values fall back to 'system'."""
    if raw not in _SETTINGS_CATEGORY_KEYS:
        return _SETTINGS_CATEGORY_DEFAULT
    return raw


def _build_settings_items(raw, display_names, *, readonly_predicate=None):
    """Flatten raw_with_docs() section→{key:value} into a UI row list.

    Args:
        raw: dict[section, dict[key, value]] — output of service.raw_with_docs().
        display_names: dict[section, dict[key, str]] — output of service.display_names().
        readonly_predicate: Optional callback (section, key) -> bool; True means
            the item is read-only.

    Returns:
        A list of dicts, each with keys: section, section_label, section_doc,
        key, name, doc, value, value_kind (bool/int/float/str/object/array),
        editable.
    """
    items = []
    for section, sec_values in raw.items():
        if section.startswith('_') or not isinstance(sec_values, dict):
            continue
        sec_names = display_names.get(section, {})
        section_label = sec_names.get('_doc', section)
        section_doc = sec_values.get('_doc', '') if isinstance(sec_values, dict) else ''
        # Sort: by display_names order (predefined order); keys not in display_names appended after
        ordered_keys = [k for k in sec_names.keys() if k != '_doc']
        extra_keys = [k for k in sec_values.keys()
                      if k != '_doc' and k not in ordered_keys]
        for key in ordered_keys + extra_keys:
            if key.startswith('_'):
                continue
            if key not in sec_values:
                continue  # Skip keys present in display_names but missing from JSON
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
                'doc': '',  # Per-key _doc not maintained in JSON for now (_doc is section-level)
                'value': value,
                'value_kind': value_kind,
                'editable': editable,
            })
    return items


def _classify_value_kind(value):
    """Map a Python value to a UI type tag (bool/int/float/str/object/array)."""
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
    """Group flat items by section for template card rendering.

    Returns [{'section': 'thresholds', 'label': '...', 'doc': '...', 'items': [...]}, ...]
    Order: preserves first-occurrence order (dict maintains insertion order).
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
    """System settings page (GET).

    GET ?category=system|theme|calibration switches the three categories
    (default system). The first two categories can render even when the Container
    is not ready (no Container dependency; calibration reads files directly, so
    the three-state gate is relaxed: only falls back to placeholder page on
    service exceptions).
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

    # Three category tabs (active marks the current selection)
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
        # For JS: set of editable items per row (for batch collecting changes)
        'editable_json': json.dumps(
            [{'section': it['section'], 'key': it['key']}
             for grp in groups for it in grp['items'] if it['editable']],
            ensure_ascii=False,
        ),
    })


def _build_calibration_groups():
    """Read channel_calibration.json raw text and build read-only display groups.

    Each record is {channel: {flip, score_type, threshold, threshold_name, ...}},
    UI expands one row per channel listing key fields (threshold/score_type/flip/threshold_name).
    Nested array fields (freq_band_mean/std) are collapsed to "<N values>" summaries, not expanded.
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
        # Treat each channel as a section, with the cfg fields as items.
        sec_items = []
        for key, value in cfg.items():
            if key.startswith('_'):
                continue
            value_kind = _classify_value_kind(value)
            # Collapse array fields into a summary placeholder.
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
                'name': key,  # Calibration field names displayed as-is (no Chinese mapping yet)
                'doc': '',
                'value': display_value,
                'value_kind': value_kind,
                'editable': False,  # All read-only
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
    """Save a single config item (POST, super-admin only).

    Input JSON: {category: 'system'|'theme', section: str, key: str, value: any}
    Output JSON: {status: 'ok', old, new} or {status: 'error', message}
    The calibration category directly returns 403 (read-only).
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
# Page 4: Alert and warning management (measured alerts only; predicted warnings on dashboard)
# ════════════════════════════════════════════════════════════════════════════
# Requirements doc (admin section "Alert and warning management (incl. warnings)"):
#   Each row shows id, alert or warning type, sensor name, telemetry value,
#   anomaly score, alert time (UTC), LLM diagnosis status, human diagnosis
#   status, comprehensive status, action (click drawer to show waveform /
#   description / LLM / annotation).
#   Top bar: Add / Move to recycle bin / Delete / LLM diagnosis / Human
#   annotation / Export (all support batch).
#
# Decisions (confirmed):
#   - This page only manages measured alerts (persisted in SQLite alert_records)
#   - Predicted warnings are already shown on the dashboard, not duplicated here
#   - The list is not real-time (requirements doc: "the interface does not
#     update in real-time; refresh the page to fetch new data")

# Filter parameter whitelist and defaults (aligned with requirements-doc admin
# section L97 column definitions: type / sensor name / LLM diagnosis / human
# diagnosis / comprehensive status; plus time window, but without exposing
# the database status field)
_ALERT_FILTER_DEFAULTS = {
    'channel': None,        # str | None — sensor channelName (UI label "传感器名称")
    'alert_type': None,     # 'measured'|'predicted'|'joint' | None
    'llm_verdict': None,    # 'real'|'false_alarm'|'uncertain'|'' | None (empty = undiagnosed)
    'human_verdict': None,  # same as llm_verdict
    'verdict': None,        # 'real'|'false_alarm'|'uncertain' (comprehensive: either human/LLM match)
    'start_ts': None,       # float | None
    'end_ts': None,         # float | None
}
_ALERT_LIMIT_DEFAULT = 20
_ALERT_LIMIT_MAX = 1000

# LLM diagnosis async thread pool (request threads not blocked; progress via alert_diagnose_status_api)
# Module-level singleton: avoids creating a new pool per request
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
_diagnose_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='phm-diagnose')
# Simple progress tracking (started/done/total/errors), separate from DiagnosisService.auto_status
# Only tracks web-triggered batch diagnosis progress
_diagnose_progress: dict = {'running': False, 'done': 0, 'total': 0,
                             'errors': 0, 'started_at': 0.0, 'finished_at': 0.0}


def _parse_alert_filters(get_params):
    """Parse filter conditions from GET parameters (aligned with the
    requirements-doc admin section L97 column definitions).

    Returns a dict whose keys match SQLiteStore.query_alerts_filtered. Invalid
    values fall back to None.

    Adds llm_verdict / human_verdict individual filtering (the requirements-doc
    admin-section columns "LLM diagnosis status" and "Human diagnosis status"
    filter independently); retains ``verdict`` for comprehensive matching
    (either human or LLM match).
    """
    out = dict(_ALERT_FILTER_DEFAULTS)

    channel = get_params.get('channel')
    if channel and isinstance(channel, str) and channel.strip():
        out['channel'] = channel.strip()[:64]  # Length limit to prevent injection

    alert_type = get_params.get('alert_type')
    if alert_type in ('measured', 'predicted', 'joint'):
        out['alert_type'] = alert_type

    # LLM diagnosis filter: supports real/false_alarm/uncertain (three diagnosed states) + 'none' (undiagnosed)
    llm_v = get_params.get('llm_verdict')
    if llm_v in ('real', 'false_alarm', 'uncertain', 'none'):
        out['llm_verdict'] = llm_v

    # Human diagnosis filter: same as above
    human_v = get_params.get('human_verdict')
    if human_v in ('real', 'false_alarm', 'uncertain', 'none'):
        out['human_verdict'] = human_v

    # Comprehensive status (retain verdict field: either human/LLM match)
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
    """Parse an ISO 8601 string or numeric string into a Unix timestamp (float).

    Supports 'YYYY-MM-DD' / 'YYYY-MM-DDTHH:MM:SS' / 'YYYY-MM-DD HH:MM:SS' / 1234567890.0.
    Returns None on failure.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    # First try pure numeric (Unix timestamp)
    try:
        return float(s)
    except ValueError:
        pass
    # ISO 8601 (space-separated also supported)
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return _dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def _parse_alert_limit(raw):
    """Parse limit parameter, clamped to [1, 1000], default 50."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _ALERT_LIMIT_DEFAULT
    return max(1, min(n, _ALERT_LIMIT_MAX))


_ALERT_PAGE_DEFAULT = 1
_ALERT_PAGE_MAX = 100000  # Upper limit to prevent abuse (100k pages x 50/page = 5M rows, far beyond actual)

# Pagination bar "items per page" dropdown candidate values (structured: single source of truth, frontend no hardcoding)
_ALERT_PAGE_SIZE_OPTIONS = [20, 50, 100, 200]


def _parse_alert_page(raw):
    """Parse page parameter, clamped to [1, _ALERT_PAGE_MAX], default 1."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _ALERT_PAGE_DEFAULT
    return max(1, min(n, _ALERT_PAGE_MAX))


def _build_page_range(page: int, total_pages: int, *, window: int = 2) -> list:
    """Build pagination page number list (current page +/- window pages + first/last + ellipsis).

    Returns list[int | str], where '..' represents an ellipsis. Example: page=5, total=10:
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
    # Dedup (page=1 when left=1 may duplicate the first entry)
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
    """Alert and warning management page (GET, measured alert list).

    GET parameters (all optional): channel / alert_type / status / verdict /
    start_ts / end_ts / limit / page. Timestamps support ISO 8601 or Unix seconds.
    When the Container is not ready, renders a placeholder page.
    """
    filters = _parse_alert_filters(request.GET)
    limit = _parse_alert_limit(request.GET.get('limit'))
    page = _parse_alert_page(request.GET.get('page'))

    c, err = _container_or_error(request)
    if c is None:
        return _render_state_page(request, err, '告警和预警管理')

    # Total row count (for pagination; same filter set as query_alerts_filtered).
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

    # Compute pagination: clamp page to the last page when it exceeds total_pages.
    total_pages = max(1, math.ceil(total_count / limit)) if total_count > 0 else 1
    if page > total_pages:
        page = total_pages
    offset = (page - 1) * limit

    # Fetch alerts (query_alerts_filtered already orders by created_at DESC).
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

    # Sensor metadata (name + unit).
    sensor_meta = _build_sensor_meta(getattr(c, 'config', None))

    # Decorate each row with badge classes + sensor name + unit.
    decorated = []
    for r in rows:
        item = dict(r)
        item['alert_type_badge'] = _alert_type_badge(r.get('alert_type'))
        item['llm_verdict_badge'] = _verdict_badge(r.get('llm_verdict'))
        item['human_verdict_badge'] = _verdict_badge(r.get('human_verdict'))
        item['final_status_badge'] = _final_status_badge(r.get('final_status'))
        # Chinese labels (consistent with the filter bars; avoid showing raw
        # English values in the data rows).
        item['alert_type_label'] = _label(_ALERT_TYPE_LABEL, r.get('alert_type'))
        item['llm_verdict_label'] = _label(_VERDICT_LABEL, r.get('llm_verdict')) if r.get('llm_verdict') else '未诊断'
        item['human_verdict_label'] = _label(_VERDICT_LABEL, r.get('human_verdict')) if r.get('human_verdict') else '未标注'
        item['final_status_label'] = _label(_STATUS_LABEL, r.get('final_status'))
        # raw_snapshot tail point = telemetry value.
        raw_value = None
        snap = r.get('raw_snapshot')
        if isinstance(snap, list) and snap:
            last = snap[-1]
            if isinstance(last, (int, float)):
                raw_value = float(last)
        item['raw_value'] = raw_value
        # Sensor name + unit.
        meta = sensor_meta.get(r.get('channel'))
        item['sensor_name'] = meta['sensor_name'] if meta else (r.get('channel') or '—')
        item['unit'] = meta['unit'] if meta else ''
        decorated.append(item)

    is_superuser = request.user.is_authenticated and request.user.is_superuser

    # Echo current filter state back to the template (form controls show current values).
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
        # Pagination
        'total_count': total_count,
        'page': page,
        'total_pages': total_pages,
        'page_range': _build_page_range(page, total_pages),
        # Page-size candidates (for the pagination-bar dropdown)
        'page_size_options': _ALERT_PAGE_SIZE_OPTIONS,
        # AJAX endpoints
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
    """Alert detail (for the drawer).

    Returns the full data of a single alert: raw_snapshot / score_snapshot /
    description / LLM diagnosis text (fetched from the DiagnosisService cache
    or generated on demand) / current verdict.
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

    # Query by id directly via SQLiteStore.get_alert_by_id (whitelisted param).
    # Older versions lack get_alert_by_id, so fall back to filtering
    # query_alerts_filtered client-side.
    try:
        row = c.sqlite.get_alert_by_id(aid)
    except AttributeError:
        # Legacy fallback: query_alerts_filtered without an id parameter,
        # then filter the whole result set client-side.
        all_rows = c.sqlite.query_alerts_filtered(limit=1000)
        row = next((r for r in all_rows if r.get('id') == aid), None)
    except Exception as e:
        logger.warning("alert_detail query failed: %s", e, exc_info=True)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    if not row:
        return JsonResponse({'status': 'error', 'message': f'告警 {aid} 不存在'},
                            status=404)

    # LLM diagnosis text: read the cache from diagnosis_records.
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

    # Sensor metadata (name + unit).
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
    """Batch human annotation (real / false_alarm / uncertain). Super-admin only.

    Input JSON: ``{ids: [int], verdict: 'real'|'false_alarm'|'uncertain'}``.
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
    """Batch soft-delete (move to recycle bin). Super-admin only.

    Input JSON: ``{ids: [int]}``.
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
    """Trigger LLM diagnosis (staff-accessible, read-only semantics).
    Single or batch.

    Input JSON: ``{ids: [int], force_refresh?: bool}``.
    Behaviour: runs DiagnosisService.diagnose(...) in a loop inside a thread
    pool; progress is reported via the status endpoint.
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

    # A diagnosis task is already running: reject to avoid concurrent runs.
    if _diagnose_progress.get('running'):
        return JsonResponse({'status': 'error',
                             'message': '已有诊断任务在跑，请等结束后再触发'},
                            status=409)

    # Collect the (channel, alert_type, alert_ts) triple for each target alert.
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

    # Initialise progress and submit the batch to the thread pool.
    _diagnose_progress.update(
        running=True, done=0, total=len(targets), errors=0,
        started_at=_time.time(), finished_at=0.0,
    )
    _diagnose_executor.submit(_run_diagnose_batch, c, targets, force_refresh)

    return JsonResponse({'status': 'ok', 'started': True, 'total': len(targets)})


def _run_diagnose_batch(container, targets, force_refresh):
    """Batch diagnosis worker (executed inside the thread pool).

    Calls DiagnosisService.diagnose(...) directly — it has its own internal
    cache (repeated clicks do not re-invoke the LLM unless
    ``force_refresh=True``).
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
    """Query diagnosis progress. Staff-accessible."""
    return JsonResponse({'status': 'ok', 'progress': dict(_diagnose_progress)})


@staff_member_required
@require_http_methods(['POST'])
def alert_diagnose_one_api(request, alert_id):
    """Single synchronous diagnosis (used by the drawer's
    "Diagnose / Re-diagnose" button).

    Unlike the batch-asynchronous ``alert_diagnose_api``, this calls
    ``DiagnosisService.diagnose()`` directly (synchronous return), suitable
    for single-item immediate feedback. DiagnosisService has its own internal
    cache (when ``force_refresh=False`` a cache hit returns immediately) and
    writes ``llm_verdict`` back to ``alert_records`` automatically.

    Input JSON: ``{force_refresh?: bool}``.
    Returns: ``{status, diagnosis_text, llm_verdict, error, elapsed_sec, cached}``.
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
    force_refresh = bool(body.get('force_refresh', True))  # Single-item default forces refresh (user-initiated).

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)

    # Fetch the alert's (channel, alert_type, alert_ts) triple.
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

    # Re-read the latest row to pick up the llm_verdict / final_status that
    # DiagnosisService wrote back.
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
    """Export alerts as CSV / JSON.

    GET parameters:
      - format: ``'csv'`` (default, UTF-8 BOM) / ``'json'``
      - ids: comma-separated (optional; export only these ids; otherwise apply
        the current filters)
      - Other filter params are the same as ``alert_view``
        (channel/alert_type/status/verdict/time window)
    """
    fmt = request.GET.get('format', 'csv').lower()
    if fmt not in ('csv', 'json'):
        fmt = 'csv'

    c, err = _container_or_error(request)
    if c is None:
        return JsonResponse({'status': 'error',
                             'message': 'PHM 服务未就绪：{}'.format(err.get('phm_state'))},
                            status=503)

    # Prefer exporting by ids; otherwise apply the filter params (same as the list page).
    ids_raw = request.GET.get('ids')
    ids = _parse_id_list(ids_raw) if ids_raw else []
    if ids:
        # Query by id list (no filters).
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
        # Export relaxes the limit cap to 100k (requirements doc L114: up to
        # 100k rows per channel).
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

    # Serialise into 5 columns (requirements doc L114):
    # channel/timestamp/raw_value/anomaly_score/received_at_iso.
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

    # CSV (UTF-8 BOM, opens directly in Excel).
    import csv
    import io as _io

    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['channel', 'timestamp', 'raw_value', 'anomaly_score',
                     'received_at_iso'])
    for row in serialised:
        writer.writerow([row['channel'], row['timestamp'], row['raw_value'],
                         row['anomaly_score'], row['received_at_iso']])

    # Streaming response (large-file friendly).
    def _stream():
        yield b'\xef\xbb\xbf'  # UTF-8 BOM
        yield buf.getvalue().encode('utf-8')

    resp = StreamingHttpResponse(_stream(), content_type='text/csv; charset=utf-8')
    resp['Content-Disposition'] = 'attachment; filename="phm_alerts.csv"'
    return resp


@staff_member_required
@require_http_methods(['POST'])
def alert_create_api(request):
    """Manually create an alert (for missed detections). Super-admin only.

    Input JSON: ``{channel: str, score: float, message?: str,
    created_at?: float (ISO string or Unix seconds),
    raw_snapshot?: list[float], score_snapshot?: list[float]}``.
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
# Page 6: Device-tree management (left tree + right edit panel + 3 drag-drop
#         semantics + cycle prevention)
# ════════════════════════════════════════════════════════════════════════════
# Requirements doc (admin section "Device-tree management"):
#   Left tree (create folder / sensor / 3 drag-drop semantics + cycle
#   prevention) + right edit panel (name / health-calc method / data-source
#   dropdown / transfer-block size / description / @ command / apply / delete)
#   + special-sensor * marker + excluded from carousel.
#
# Decisions (confirmed):
#   - Model binding stays implicit via @ commands (schema unchanged)
#   - Service layer is untouched: ConfigService.save already handles empty-tree
#     protection + duplicate sourceId validation + TCP push
#   - Drag-drop cycle prevention is implemented in front-end JS (DFS on the
#     ancestor chain); the backend does not redo it

# space_daq_channels.json location (same directory as system_config).
_SPACE_CHANNELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "space_daq_channels.json",
)


def _load_space_channels():
    """Read space_daq_channels.json for the device-tree "data-source dropdown".

    Returns ``[{id, source_id, label, enabled}]``. Returns ``[]`` if the file
    is missing.
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
    """Recursively mark sensors with ``_special`` (has an @rul command or
    isSpecial=true).

    Returns a new tree (deep-copied to avoid mutating service-held state).
    The front-end uses this flag to render the ``*`` marker.
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
    """Device-tree management page (GET).

    When the Container is not ready, renders a placeholder page. When ready,
    renders the left tree + an empty right panel; front-end JS handles node
    selection, editing, drag-drop, cycle prevention, and saving.
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
    """Data source for the "data-source dropdown" (GET).

    Returns the contents of space_daq_channels.json (same source as the
    ``channels_json`` injected by ``device_tree_view``). Exposed as a separate
    endpoint so the front-end can refresh dynamically (after the admin edits
    space_daq_channels.json, no page reload is needed).
    """
    return JsonResponse({
        'status': 'ok',
        'channels': _load_space_channels(),
    })


@staff_member_required
@require_http_methods(['POST'])
def device_tree_save_api(request):
    """Save the whole tree (POST, super-admin only).

    Input JSON: the full body of device_config.json (with ``device_tree`` +
    ``aggregation_strategy``). You may also pass only
    ``{device_tree: [...]}``; when ``aggregation_strategy`` is omitted, the
    previous value is preserved.
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
