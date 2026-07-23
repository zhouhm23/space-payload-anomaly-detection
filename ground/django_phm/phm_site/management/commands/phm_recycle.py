"""Recycle bin management command.

Provides an Agent/CLI-friendly recycle bin operation channel (the path
parallel to the HTTP API, per AGENTS.md §Structured Design Spec item 5
"Agent-friendly dual channel").

Subcommands:
  --list   --table alerts|detections|diagnoses [--limit N] [--format json|text]
  --restore --table alerts|detections|diagnoses --ids 1,2,3
  --purge  --table alerts|detections|diagnoses --ids 1,2,3

Directly calls services_bridge.get_container().sqlite. Does not duplicate
business logic.

Note: the CLI does not go through Django auth by default, so it does **not**
verify super-admin status. In production, ensure this is only run in a
trusted environment (consistent with the usual manage.py assumptions).

Examples:
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
        # Mutually exclusive action group (pick one of three)
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
        # Ensure PHM service is ready first (CLI starts services_bridge directly,
        # bypassing WSGI middleware). The CLI is a separate process and cannot
        # share the runserver Container, so it must start its own.
        # (Background initialisation includes model loading; wait up to 60 s)
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

    # ── Sub-operations ──────────────────────────────────────────────────

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
            # Text mode: compact table
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
