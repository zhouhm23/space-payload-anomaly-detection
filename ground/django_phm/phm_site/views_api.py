"""DRF 新接口视图（/api/v2/）。

需求书 §前台监控大屏 所需的数据源 API。
所有需要 Container 的端点先检查 services_bridge.get_state()，
未就绪时返回 503 + 状态提示，避免误导前端报 500。

端点清单（v1.1 第一轮 1b）：
- GET /api/v2/ping/                健康探针（纯 Django 进程）
- GET /api/v2/startup-status/      Container 三态探针（前端轮询）
- GET /api/v2/theme/               前台主题
- GET /api/v2/system-info/         顶栏系统信息
- GET /api/v2/device-tree/         设备树（含健康值聚合）
- GET /api/v2/window/              遥测窗口（raw+pred 同行）
- GET /api/v2/alerts/              实测告警列表（含 verdict 四维度）
- GET /api/v2/warnings/            预测预警列表
- GET /api/v2/rul/                 RUL 退化预测（特殊传感器）
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


# ── 通用工具 ─────────────────────────────────────────────────────────────────
def _container_or_503():
    """获取 Container，未就绪时返回 (None, 503 Response)。"""
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


# ── 系统类（无需 Container） ─────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def ping_view(request):
    """健康探针（纯 Django 进程，不检查 Container）。"""
    return Response({
        'status': 'ok',
        'service': 'phm-ground',
        'version': 'v1.1',
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def startup_status_view(request):
    """启动状态探针（前端轮询用）。

    返回 services_bridge 三态：idle / initializing / ready / failed。
    前端等到 state=ready 才开始拉业务数据。
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
    """前台主题配置（动态可刷，首屏仍走 context_processor）。"""
    return Response(get_theme().as_dict())


# ── 顶栏 ────────────────────────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def system_info_view(request):
    """系统信息（顶栏用）：天地链接状态、延迟、UTC 时间、系统名称。

    天地延迟 = TCP 往返时延（services_bridge 测量 poll 耗时），
    本地测试应接近 0；真实卫星链路为秒级。
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

    # 从 services_bridge 拉链路 RTT 统计
    link = services_bridge.get_link_status()
    info['link_status'] = link['status']
    info['link_latency_ms'] = round(link['rtt_ms'], 1) if link['rtt_ms'] is not None else None
    return Response(info)


# ── 设备树（含健康值聚合） ──────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def device_tree_view(request):
    """设备树（需求书 §左设备树区）。

    返回 ConfigService.load() + HealthService 健康值聚合（系统+逐通道+文件夹）。
    特殊传感器（非一维数据源）名称后端加 `*` 标注，前端不参与轮播。
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


# ── 遥测窗口（图表主数据源） ────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def window_view(request):
    """实时遥测窗口（需求书 §中图表区 主数据源）。

    数据源：RingBuffer（实时，无 flush 延迟）+ WarningService._latest_predict_scores（实时 pred）。
    比 SQLite query_window 更新鲜，适合实时大屏 2s 刷新。

    Query params:
        channel: 通道名（必填）
        count:   行数（默认 512）

    返回结构（raw + pred 合并对齐到同一时间戳）：
        {
          channel, count, threshold,
          data: [{timestamp, raw_value, anomaly_score,
                  predicted_value, predicted_anomaly_score}, ...],
          predict_window: {start, end, scores: [...]}  # 预测段元信息
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
        # 数据源：SQLite query_window（已做 raw+pred UPSERT 合并，时间戳对齐）
        # 主分支策略：默认拉 2048 行（4× 可视区 512），前端按需截取可视子段
        # 这样能覆盖到 pred 数据（pred 落后 raw 几秒，必须拉足够长窗口）
        if count < 2048:
            count = 2048
        sqlite_result = c.sqlite.query_window(channel, count=count)
        data = sqlite_result.get('data', [])

        # 阈值（从设备树 sensor 配置取，fallback 默认 0.5）
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


# ── 全通道告警点图数据 ──────────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def alert_points_view(request):
    """全通道告警点图（需求书 §全通道告警点图区）。

    返回所有通道在统一时间轴上的告警红点（实测）和预警黄点（预测）。
    每个点含 (channel, timestamp, score, type)。
    """
    c, err_resp = _container_or_503()
    if err_resp is not None:
        return err_resp

    try:
        # 实测告警（SQLite 历史最近 50 条）
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

        # 预测预警（内存 WarningStore）
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

        # 全通道列表（用于 Y 轴分类）
        channels = sorted(set(
            [p['channel'] for p in red_points + yellow_points if p.get('channel')]
        ))

        return Response({
            'red_points': red_points,      # 实测告警
            'yellow_points': yellow_points, # 预测预警
            'channels': channels,
        })
    except Exception as e:
        return Response(
            {'detail': f'查询告警点失败: {e}'},
            status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── 实测告警 + 预测预警 ─────────────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def alerts_view(request):
    """实测告警列表（需求书 §右详情区下半部分）。

    对齐主分支风格：从内存 AlertStore 取实时告警（含 raw_window/score_window 快照），
    合并 SQLite 的 verdict 四维度。返回字段精简，前端直接渲染。

    每条告警字段：
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
        # 从内存 AlertStore 取实时告警（AlertPacket 结构，含 raw_window/score_window）
        alerts = c.alert_service.list(limit)
        # 合并 SQLite 的 verdict 四维度（key=channel+time）
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
    """预测预警列表（需求书 §右详情区下半部分，黄色预警）。

    返回 WarningStore 中的 pending/confirmed/false/unverifiable 预警。
    """
    c, err_resp = _container_or_503()
    if err_resp is not None:
        return err_resp

    try:
        limit = max(1, min(int(request.GET.get('limit', '20')), 100))
    except ValueError:
        limit = 20

    try:
        # WarningStore.recent(limit) 返回 list[dict]（WarningEntry.to_dict() 已序列化）
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


# ── RUL 退化预测（特殊传感器） ──────────────────────────────────────────────
@api_view(['GET'])
@permission_classes([AllowAny])
def rul_view(request):
    """RUL 退化预测（需求书 §特殊传感器）。

    标记为 @rul:fd001 的特殊传感器才返回数据。
    无标记通道或资产缺失时返回 503。
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
