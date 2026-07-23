"""DRF v2 views (/api/v2/).

The data-source APIs needed by the spec front-end monitor dashboard.
Every endpoint that needs the Container first checks
services_bridge.get_state(); when not ready it returns 503 + a status hint,
so the front-end is never misled into a 500.

Endpoint list (round 1b of v1.1):
- GET /api/v2/ping/                health probe (pure Django process)
- GET /api/v2/startup-status/      Container three-state probe (front-end poll)
- GET /api/v2/theme/               front-end theme
- GET /api/v2/system-info/         top-bar system info
- GET /api/v2/device-tree/         device tree (with health aggregation)
- GET /api/v2/window/              telemetry window (raw+pred per row)
- GET /api/v2/alerts/              measured-alert list (with the four-dimension verdict)
- GET /api/v2/warnings/            predicted-warning list
- GET /api/v2/rul/                 RUL degradation prediction (special sensors)
"""
from __future__ import annotations

from datetime import datetime, timezone
import time

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status as http_status

from phm.services.theme_service import get_theme
from . import services_bridge


# ── Common helpers ──────────────────────────────────────────────────────────
def _container_or_503():
    """Get the Container; return (None, 503 Response) when not ready."""
    state = services_bridge.get_state()
    if state == 'failed':
        err = services_bridge.get_init_error()
        return None, Response(
            {'detail': f'PHM 初始化失败: {err}', 'state': state},
            status=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    if state != 'ready':
        return None, Response(
            {'detail': f'PHM 初始化中，请稍候（state={state}）', 'state': state},
            status=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:
        return services_bridge.get_container(), None
    except RuntimeError as e:
        return None, Response(
            {'detail': str(e), 'state': state},
            status=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        )


# ── System endpoints (no Container needed) ──────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def ping_view(request):
    """Health probe (pure Django process; does not check the Container)."""
    return Response({
        'status': 'ok',
        'service': 'phm-ground',
        'version': 'v1.1',
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def startup_status_view(request):
    """Startup-state probe (for the front-end poll).

    Returns the services_bridge three states: idle / initializing / ready / failed.
    The front-end waits until state=ready before fetching business data.
    """
    state = services_bridge.get_state()
    payload = {
        'state': state,
        'ready': state == 'ready',
    }
    if state == 'failed':
        payload['error'] = services_bridge.get_init_error()
    return Response(payload)


@api_view(['GET'])
@permission_classes([AllowAny])
def theme_view(request):
    """Front-end theme config (dynamic, refreshable; first paint still goes through the context processor)."""
    return Response(get_theme().as_dict())


# ── Top bar ─────────────────────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def system_info_view(request):
    """System info (for the top bar): space-ground link status, latency, UTC time, system name.

    Space-ground latency = TCP round-trip (services_bridge measures the poll
    cost); it should be near 0 in local tests and on the order of seconds on a
    real satellite link.
    """
    info = {
        'system_title': '空间有效载荷天地协同健康管理系统',
        'utc_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'utc_epoch': time.time(),
        'link_status': 'initializing',
        'link_latency_ms': None,
    }

    state = services_bridge.get_state()
    if state != 'ready':
        info['link_status'] = 'initializing'
        info['note'] = 'PHM 正在初始化'
        return Response(info)

    # Pull the link RTT statistics from services_bridge
    link = services_bridge.get_link_status()
    info['link_status'] = link['status']
    info['link_latency_ms'] = round(link['rtt_ms'], 1) if link['rtt_ms'] is not None else None
    return Response(info)


# ── Device tree (with health aggregation) ───────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def device_tree_view(request):
    """Device tree (spec left device-tree panel).

    Returns ConfigService.load() + HealthService health aggregation
    (system + per-channel + per-folder). Special sensors (non-1-D sources) get
    a `*` suffix on their name on the back end and are excluded from the
    front-end carousel.
    """
    c, err_resp = _container_or_503()
    if err_resp is not None:
        return err_resp

    try:
        config_data = c.config.load()
        health = c.health.system_health()
        return Response({
            'device_tree': config_data.get('device_tree', []),
            'aggregation_strategy': config_data.get('aggregation_strategy', 'min'),
            'health': health,
        })
    except Exception as e:
        return Response(
            {'detail': f'读取设备树失败: {e}'},
            status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── Telemetry window (the chart's main data source) ─────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def window_view(request):
    """Live telemetry window (the main data source for the spec centre-chart panel).

    Data source: RingBuffer (live, no flush latency) +
    WarningService._latest_predict_scores (live pred). Fresher than
    SQLite query_window, suitable for a 2s live-monitor refresh.

    Query params:
        channel: channel name (required)
        count:   row count (default 512)

    Return shape (raw + pred merged and aligned on the same timestamp):
        {
          channel, count, threshold,
          data: [{timestamp, raw_value, anomaly_score,
                  predicted_value, predicted_anomaly_score}, ...],
          predict_window: {start, end, scores: [...]}  # prediction-segment metadata
        }
    """
    c, err_resp = _container_or_503()
    if err_resp is not None:
        return err_resp

    channel = request.GET.get('channel', '').strip()
    if not channel:
        return Response(
            {'detail': '缺少 channel 参数'},
            status=http_status.HTTP_400_BAD_REQUEST,
        )
    try:
        count = max(1, min(int(request.GET.get('count', '512')), 10000))
    except ValueError:
        count = 512

    try:
        # Data source: SQLite query_window (raw+pred UPSERT-merged, timestamps aligned).
        # Main-branch policy: pull 2048 rows by default (4× the 512-row viewport);
        # the front-end slices the visible sub-window as needed. This is necessary
        # to cover the pred data (pred lags raw by a few seconds, so the window
        # must be long enough).
        if count < 2048:
            count = 2048
        sqlite_result = c.sqlite.query_window(channel, count=count)
        data = sqlite_result.get('data', [])

        # Threshold (from the device-tree sensor config, fallback default 0.5)
        threshold = 0.5
        try:
            from phm.services.tree_utils import get_flat_sensors
            cfg = c.config.load()
            for s in get_flat_sensors(cfg.get('device_tree', [])):
                if s.get('channelName') == channel and s.get('threshold') is not None:
                    threshold = s['threshold']
                    break
        except Exception:
            pass

        return Response({
            'channel': channel,
            'count': len(data),
            'threshold': threshold,
            'gaps': sqlite_result.get('gaps', []),
            'gap_threshold': sqlite_result.get('gap_threshold', 5.12),
            'data': data,
        })
    except Exception as e:
        return Response(
            {'detail': f'查询窗口失败: {e}'},
            status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── All-channel alert-point map data ────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def alert_points_view(request):
    """All-channel alert-point map (spec all-channel alert-point panel).

    Returns every channel's alert red dots (measured) and warning yellow dots
    (predicted) on a unified time axis. Each point carries
    (channel, timestamp, score, type).
    """
    c, err_resp = _container_or_503()
    if err_resp is not None:
        return err_resp

    try:
        # Measured alerts (most recent 50 from SQLite)
        alerts = c.sqlite.query_alerts(limit=50)
        red_points = [
            {
                'channel': a.get('channel'),
                'timestamp': a.get('created_at'),
                'score': a.get('score'),
                'type': 'measured',
            }
            for a in alerts
            if a.get('alert_type') == 'measured'
        ]

        # Predicted warnings (in-memory WarningStore)
        yellow_points = []
        try:
            warnings_list = c.warnings.recent(limit=50)
            for w in warnings_list:
                yellow_points.append({
                    'channel': w.get('channel'),
                    'timestamp': w.get('created_at') or w.get('start_ts'),
                    'score': w.get('max_score') or w.get('score'),
                    'type': 'predicted',
                })
        except Exception:
            pass

        # All-channel list (for Y-axis categories)
        channels = sorted(set(
            [p['channel'] for p in red_points + yellow_points if p.get('channel')]
        ))

        return Response({
            'red_points': red_points,      # measured alerts
            'yellow_points': yellow_points, # predicted warnings
            'channels': channels,
        })
    except Exception as e:
        return Response(
            {'detail': f'查询告警点失败: {e}'},
            status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── Measured alerts + predicted warnings ────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def alerts_view(request):
    """Measured-alert list (spec right-detail panel, lower half).

    Aligned with the main-branch style: pulls live alerts from the in-memory
    AlertStore (with raw_window/score_window snapshots) and merges the
    four-dimension verdict from SQLite. The returned fields are trimmed so the
    front-end can render directly.

    Per-alert fields:
      channel, time, score, message, raw_window (list), score_window (list),
      human_verdict, llm_verdict, final_status
    """
    c, err_resp = _container_or_503()
    if err_resp is not None:
        return err_resp

    try:
        limit = max(1, min(int(request.GET.get('limit', '20')), 100))
    except ValueError:
        limit = 20

    try:
        # Live alerts from the in-memory AlertStore (AlertPacket structure, with raw_window/score_window)
        alerts = c.alert_service.list(limit)
        # Merge the four-dimension verdict from SQLite (key=channel+time)
        try:
            db_rows = c.sqlite.query_alerts(limit=limit * 2)
            verdict_map = {}
            for r in db_rows:
                key = (r.get('channel'), round(float(r.get('created_at', 0)), 3))
                verdict_map[key] = {
                    'human_verdict': r.get('human_verdict'),
                    'llm_verdict': r.get('llm_verdict'),
                    'final_status': r.get('final_status'),
                }
            for a in alerts:
                key = (a.get('channel'), round(float(a.get('time', 0)), 3))
                v = verdict_map.get(key)
                if v:
                    a['human_verdict'] = v['human_verdict']
                    a['llm_verdict'] = v['llm_verdict']
                    a['final_status'] = v['final_status']
        except Exception:
            pass

        return Response({
            'alerts': alerts,
            'threshold': c.alert_service.threshold,
        })
    except Exception as e:
        return Response(
            {'detail': f'查询告警失败: {e}'},
            status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def warnings_view(request):
    """Predicted-warning list (spec right-detail panel, lower half; yellow warnings).

    Returns the pending/confirmed/false/unverifiable warnings from WarningStore.
    """
    c, err_resp = _container_or_503()
    if err_resp is not None:
        return err_resp

    try:
        limit = max(1, min(int(request.GET.get('limit', '20')), 100))
    except ValueError:
        limit = 20

    try:
        # WarningStore.recent(limit) returns list[dict] (WarningEntry.to_dict() already serialised)
        warnings_list = c.warnings.recent(limit=limit) if hasattr(c.warnings, 'recent') else []
        return Response({
            'warnings': warnings_list,
            'count': len(warnings_list),
        })
    except Exception as e:
        return Response(
            {'detail': f'查询预警失败: {e}'},
            status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── RUL degradation prediction (special sensors) ────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def rul_view(request):
    """RUL degradation prediction (spec special-sensor panel).

    Only special sensors tagged @rul:fd001 return data. Untagged channels or
    missing assets return 503.
    """
    c, err_resp = _container_or_503()
    if err_resp is not None:
        return err_resp

    if c.rul is None:
        return Response(
            {'detail': 'RUL 服务未启用（资产缺失或无标记通道）', 'enabled': False},
            status=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    channel = request.GET.get('channel')
    try:
        if channel:
            result = c.rul.predict(channel)
            if result is None:
                return Response(
                    {'detail': f'通道 {channel} 未标记为 RUL', 'enabled': True},
                    status=http_status.HTTP_404_NOT_FOUND,
                )
            return Response(result)
        else:
            results = c.rul.predict_all()
            return Response({'channels': results, 'count': len(results)})
    except Exception as e:
        return Response(
            {'detail': f'RUL 预测失败: {e}'},
            status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
