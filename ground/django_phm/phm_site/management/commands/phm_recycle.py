""" recycle bin management command.

提供 Agent/CLI 友好的回收站操作通道（HTTP API 之外的另一条路径，
对齐 AGENTS.md §结构化设计规范 第 5 条「Agent 友好双通道」）。

子命令：
  --list   --table alerts|detections|diagnoses [--limit N] [--format json|text]
  --restore --table alerts|detections|diagnoses --ids 1,2,3
  --purge  --table alerts|detections|diagnoses --ids 1,2,3

直调 services_bridge.get_container().sqlite，不重复业务逻辑。

注意：CLI 默认不经过 Django auth，因此**不校验超管**。生产环境部署时
请确保只在受信环境运行（与 manage.py 的常规假设一致）。

示例：
  python manage.py phm_recycle --list --table alerts --format json
  python manage.py phm_recycle --restore --table alerts --ids 12,34
  python manage.py phm_recycle --purge --table diagnoses --ids 56
"""
from __future__ import annotations

import json as _json
import time

from django.core.management.base import BaseCommand, CommandError

from django.core.management.base import BaseCommand, CommandError

from phm_site import services_bridge
from phm_site.views_admin import _RECYCLE_TABLE_MAP, _parse_recycle_table, _parse_id_list


class Command(BaseCommand):
    help = "回收站：列出 / 恢复 / 永久删除已软删的告警/检测/诊断记录"

    def add_arguments(self, parser):
        # 互斥的动作组（三个三选一）
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--list', action='store_true', help='列出已软删记录')
        group.add_argument('--restore', action='store_true', help='恢复软删记录（is_deleted→0）')
        group.add_argument('--purge', action='store_true', help='永久删除（物理删除）')

        parser.add_argument(
            '--table', default='alerts',
            choices=list(_RECYCLE_TABLE_MAP.keys()),
            help='资源类型：alerts / detections / diagnoses（默认 alerts）',
        )
        parser.add_argument(
            '--ids', default='',
            help='要操作的 id 列表（逗号分隔），用于 --restore / --purge',
        )
        parser.add_argument(
            '--limit', type=int, default=200,
            help='--list 单次返回上限（默认 200，最大 1000）',
        )
        parser.add_argument(
            '--format', dest='output_format', default='text',
            choices=['text', 'json'],
            help='输出格式（默认 text）',
        )

    def handle(self, *args, **opts):
        # 先确保 PHM 服务就绪（CLI 直接启动 services_bridge，不走 WSGI 中间件）
        # CLI 是独立进程，不能共享 runserver 的 Container，需要自己启一个
        # （后台初始化含模型加载，最多等 60 秒）
        if services_bridge.get_state() == 'idle':
            self.stdout.write(self.style.WARNING(
                "PHM 服务未启动，正在后台初始化（加载模型，约需 10-30 秒）…"
            ))
            services_bridge.start()
        deadline = time.time() + 60
        while services_bridge.get_state() == 'initializing' and time.time() < deadline:
            time.sleep(1)
        state = services_bridge.get_state()
        if state != 'ready':
            err = services_bridge.get_init_error()
            raise CommandError(
                f"PHM 服务未就绪（state={state}）"
                + (f"：{err}" if err else "（超时 60s）")
            )
        c = services_bridge.get_container()
        if c is None:
            raise CommandError("PHM Container 不可用")

        table_key, sql_table, label = _parse_recycle_table(opts['table'])

        if opts['list']:
            self._do_list(c, sql_table, table_key, label, opts)
        elif opts['restore']:
            self._do_mutation(c.sqlite.restore, sql_table, table_key, label, opts, action='restore')
        elif opts['purge']:
            self._do_mutation(c.sqlite.purge_by_ids, sql_table, table_key, label, opts, action='purge')

    # ── 子操作 ──────────────────────────────────────────────────

    def _do_list(self, c, sql_table, table_key, label, opts):
        limit = max(1, min(int(opts['limit']), 1000))
        rows = c.sqlite.query_deleted(sql_table, limit=limit)
        if opts['output_format'] == 'json':
            self.stdout.write(_json.dumps({
                'table': table_key, 'label': label, 'count': len(rows), 'rows': rows,
            }, ensure_ascii=False, indent=2))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"回收站 · {label}（{table_key}）：共 {len(rows)} 条（上限 {limit}）"
            ))
            if not rows:
                return
            # 文本模式：简表
            for r in rows:
                rid = r.get('id')
                channel = r.get('channel', '—')
                ts = r.get('created_at') or r.get('timestamp') or r.get('alert_ts')
                deleted = r.get('deleted_at')
                self.stdout.write(
                    f"  id={rid}\tchannel={channel}\tts={ts}\tdeleted_at={deleted}"
                )

    def _do_mutation(self, fn, sql_table, table_key, label, opts, *, action):
        ids = _parse_id_list(opts['ids'])
        if not ids:
            raise CommandError(f"--{action} 需要 --ids（逗号分隔的正整数）")
        n = fn(sql_table, ids)
        if opts['output_format'] == 'json':
            key = 'restored' if action == 'restore' else 'purged'
            self.stdout.write(_json.dumps({
                'status': 'ok', 'table': table_key, 'requested': len(ids), key: n,
            }, ensure_ascii=False))
        else:
            verb = '恢复' if action == 'restore' else '永久删除'
            self.stdout.write(self.style.SUCCESS(
                f"已{verb} {n}/{len(ids)} 条{label}（table={table_key}）"
            ))
