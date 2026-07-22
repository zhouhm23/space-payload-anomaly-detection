"""Device tree management command.

Agent 友好双通道：与「设备树管理」HTTP API 并行。

子命令：
  --list       [--format json|text]            # 展示当前设备树
  --validate   --file new_config.json          # 预检（空树 / 重复 sourceId）
  --save       --file new_config.json          # 保存整树（含 TCP 推送）

直调 services_bridge.get_container().config，不重复业务逻辑。

示例：
  python manage.py phm_device_tree --list --format json
  python manage.py phm_device_tree --validate --file new_config.json
  python manage.py phm_device_tree --save --file new_config.json
"""
from __future__ import annotations

import json as _json
import time

from django.core.management.base import BaseCommand, CommandError

from phm_site import services_bridge


def _ensure_ready():
    """确保 PHM Container 就绪（CLI 直启 services_bridge）。"""
    if services_bridge.get_state() == 'idle':
        services_bridge.start()
    deadline = time.time() + 60
    while services_bridge.get_state() == 'initializing' and time.time() < deadline:
        time.sleep(1)
    state = services_bridge.get_state()
    if state != 'ready':
        err = services_bridge.get_init_error()
        raise CommandError(
            f"PHM 服务未就绪（state={state}）" +
            (f"：{err}" if err else "（超时 60s）")
        )
    return services_bridge.get_container()


def _count_sensors(tree):
    """递归统计传感器数量。"""
    n = 0
    def walk(nodes):
        nonlocal n
        for node in nodes or []:
            if isinstance(node, dict):
                if node.get('type') == 'sensor':
                    n += 1
                walk(node.get('children'))
    walk(tree)
    return n


def _find_duplicate_source_id(tree):
    """复用 ConfigService._find_duplicate_source 检测重复 sourceId。

    返回首个重复的 sourceId，或 None。
    """
    from phm.services.config_service import ConfigService
    return ConfigService._find_duplicate_source(tree)


class Command(BaseCommand):
    help = "设备树管理：列出 / 预检 / 保存"

    def add_arguments(self, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--list', action='store_true', help='列出当前设备树')
        group.add_argument('--validate', action='store_true',
                           help='预检新配置（空树/重复 sourceId）')
        group.add_argument('--save', action='store_true', help='保存整树')

        parser.add_argument('--file', default='',
                            help='--validate / --save 的 JSON 配置文件路径')
        parser.add_argument('--format', dest='output_format', default='text',
                            choices=['text', 'json'],
                            help='--list 输出格式（默认 text）')

    def handle(self, *args, **opts):
        if opts['list']:
            self._do_list(opts)
        elif opts['validate']:
            self._do_validate(opts)
        elif opts['save']:
            self._do_save(opts)

    # ── 子操作 ──────────────────────────────────────────────────

    def _do_list(self, opts):
        c = _ensure_ready()
        cfg = c.config.load()
        tree = cfg.get('device_tree', [])
        agg = cfg.get('aggregation_strategy', 'min')
        if opts['output_format'] == 'json':
            self.stdout.write(_json.dumps({
                'aggregation_strategy': agg,
                'device_tree': tree,
                'sensor_count': _count_sensors(tree),
            }, ensure_ascii=False, indent=2))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"设备树 · 聚合策略={agg} · 传感器数={_count_sensors(tree)}"
            ))
            self._print_tree(tree, depth=0)

    def _print_tree(self, nodes, depth):
        for n in nodes or []:
            if not isinstance(n, dict):
                continue
            prefix = '  ' * depth
            icon = '📁' if n.get('type') == 'folder' else '📡'
            name = n.get('name') or n.get('channelName') or n.get('sourceId') or '?'
            self.stdout.write(f"{prefix}{icon} {name} ({n.get('type')})")
            self._print_tree(n.get('children'), depth + 1)

    def _load_body(self, opts):
        """从 --file 加载 JSON body，含基础结构校验。"""
        if not opts['file']:
            raise CommandError("--validate / --save 需要 --file（JSON 文件路径）")
        try:
            with open(opts['file'], encoding='utf-8') as f:
                body = _json.load(f)
        except FileNotFoundError:
            raise CommandError(f"文件不存在：{opts['file']}")
        except _json.JSONDecodeError as e:
            raise CommandError(f"JSON 解析失败：{e}")
        if not isinstance(body, dict):
            raise CommandError("JSON 根必须是对象")
        if 'device_tree' not in body:
            # 兼容：直接传树数组（外层包装）
            if isinstance(body, list):
                body = {'device_tree': body}
            else:
                raise CommandError("JSON 必须含 device_tree 字段或本身为数组")
        return body

    def _do_validate(self, opts):
        body = self._load_body(opts)
        tree = body.get('device_tree')
        if not isinstance(tree, list):
            raise CommandError("device_tree 必须为数组")
        if not tree:
            raise CommandError("❌ 拒绝：空设备树（安全保护，会清空配置）")
        dup = _find_duplicate_source_id(tree)
        if dup:
            raise CommandError(f"❌ 拒绝：重复的 sourceId = {dup}")
        self.stdout.write(self.style.SUCCESS(
            f"✓ 预检通过 · {_count_sensors(tree)} 个传感器"
        ))

    def _do_save(self, opts):
        # 先预检
        body = self._load_body(opts)
        tree = body.get('device_tree')
        if not isinstance(tree, list) or not tree:
            raise CommandError("❌ 拒绝：空设备树（安全保护）")
        dup = _find_duplicate_source_id(tree)
        if dup:
            raise CommandError(f"❌ 拒绝：重复的 sourceId = {dup}")

        c = _ensure_ready()
        # 补 aggregation_strategy 缺省
        if 'aggregation_strategy' not in body:
            body['aggregation_strategy'] = 'min'
        result = c.config.save(body)
        if result.get('status') != 'ok':
            raise CommandError(result.get('message', '保存失败'))
        self.stdout.write(self.style.SUCCESS(
            f"✓ 设备树已保存（含 TCP 推送 best-effort）· {_count_sensors(tree)} 个传感器"
        ))
