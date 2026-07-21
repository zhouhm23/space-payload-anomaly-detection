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

import json
import logging
import os
from pathlib import Path

from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from phm.algorithm._registry import MODEL_REGISTRY, get_model_entry
from phm.services.theme_service import get_theme

from . import services_bridge

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
