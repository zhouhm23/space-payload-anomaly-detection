"""Alert management command.

Agent 友好双通道：与「告警管理」HTTP API 并行的 CLI。

子命令：
  --list      [--channel X --verdict real --status active --from 2026-07-01 --to 2026-07-21]
              [--limit N] [--format json|text]
  --annotate  --ids 1,2,3 --verdict real|false_alarm|uncertain
  --delete    --ids 1,2,3    （软删，移到回收站）
  --create    --channel C-1 --score 0.9 [--message "..."] [--ts "2026-07-21T12:00:00"]
  --export    [--ids 1,2,3 | --channel X --from ... --to ...] [--format csv|json] --out alerts.csv

直调 services_bridge.get_container().sqlite / diagnosis / DiagnosisService，
不重复业务逻辑。
"""
from __future__ import annotations

import json as _json
import time

from django.core.management.base import BaseCommand, CommandError

from phm_site import services_bridge
from phm_site.views_admin import _parse_alert_filters, _parse_id_list


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


class Command(BaseCommand):
    help = "告警管理：列表/筛选/批量标注/删除/补录/导出"

    def add_arguments(self, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--list', action='store_true', help='列出告警（支持筛选）')
        group.add_argument('--annotate', action='store_true', help='批量人工标注')
        group.add_argument('--delete', action='store_true', help='软删（移到回收站）')
        group.add_argument('--create', action='store_true', help='人工补录')
        group.add_argument('--export', action='store_true', help='导出 CSV/JSON')

        # 筛选（--list / --export 通用）
        parser.add_argument('--channel', default='', help='按通道过滤')
        parser.add_argument('--alert_type', default='',
                            choices=['', 'measured', 'predicted', 'joint'],
                            help='按类型过滤')
        parser.add_argument('--status', default='',
                            choices=['', 'active', 'pending', 'confirmed', 'false'],
                            help='按状态过滤')
        parser.add_argument('--verdict', default='',
                            choices=['', 'real', 'false_alarm', 'uncertain'],
                            help='按综合 verdict 过滤')
        parser.add_argument('--from', dest='start_ts', default='',
                            help='起始时间（ISO 字符串或 Unix 秒）')
        parser.add_argument('--to', dest='end_ts', default='',
                            help='结束时间（ISO 字符串或 Unix 秒）')
        parser.add_argument('--limit', type=int, default=50,
                            help='--list 单次返回上限（默认 50，最大 1000）')
        parser.add_argument('--page', type=int, default=1,
                            help='--list 页码（默认 1，配合 --limit 翻页）')

        # id 列表（--annotate / --delete / --export 通用）
        parser.add_argument('--ids', default='',
                            help='id 列表（逗号分隔），用于 --annotate / --delete / --export')
        parser.add_argument('--verdict_value', default='',
                            choices=['real', 'false_alarm', 'uncertain'],
                            help='--annotate 的 verdict 值')

        # --create 参数
        parser.add_argument('--score', type=float, default=None,
                            help='--create 的异常分数')
        parser.add_argument('--message', default='', help='--create 的描述')
        parser.add_argument('--ts', default='', help='--create 的告警时间（ISO 或 Unix 秒）')

        # --export / --list 输出格式
        parser.add_argument('--format', dest='output_format', default='text',
                            choices=['text', 'json', 'csv'],
                            help='输出格式（--list 默认 text；--export 默认 csv）')
        parser.add_argument('--out', default='',
                            help='--export 输出文件路径（不填则打到 stdout）')

    def handle(self, *args, **opts):
        if opts['list']:
            self._do_list(opts)
        elif opts['annotate']:
            self._do_annotate(opts)
        elif opts['delete']:
            self._do_delete(opts)
        elif opts['create']:
            self._do_create(opts)
        elif opts['export']:
            self._do_export(opts)

    # ── 子操作 ──────────────────────────────────────────────────

    def _build_filters(self, opts, *, use_ids=False):
        """从 opts 构造筛选参数（与 alert_view 共用 _parse_alert_filters）。"""
        if use_ids and opts.get('ids'):
            ids = _parse_id_list(opts['ids'])
            if ids:
                return {'_ids': ids}
        # 复用 view 层 helper
        fake_get = {
            'channel': opts.get('channel') or None,
            'alert_type': opts.get('alert_type') or None,
            'status': opts.get('status') or None,
            'verdict': opts.get('verdict') or None,
            'start_ts': opts.get('start_ts') or None,
            'end_ts': opts.get('end_ts') or None,
        }
        return _parse_alert_filters(fake_get)

    def _do_list(self, opts):
        c = _ensure_ready()
        filters = self._build_filters(opts)
        limit = max(1, min(int(opts['limit']), 1000))
        # 总数 + 分页
        total = c.sqlite.count_alerts_filtered(
            channel=filters['channel'], alert_type=filters['alert_type'],
            status=filters['status'], verdict=filters['verdict'],
            start_ts=filters['start_ts'], end_ts=filters['end_ts'],
        )
        total_pages = max(1, (total + limit - 1) // limit) if total > 0 else 1
        page = max(1, min(int(opts['page']), total_pages))
        offset = (page - 1) * limit
        rows = c.sqlite.query_alerts_filtered(
            channel=filters['channel'], alert_type=filters['alert_type'],
            status=filters['status'], verdict=filters['verdict'],
            start_ts=filters['start_ts'], end_ts=filters['end_ts'],
            limit=limit, offset=offset,
        )
        if opts['output_format'] == 'json':
            self.stdout.write(_json.dumps({
                'count': len(rows),
                'total': total,
                'page': page,
                'total_pages': total_pages,
                'limit': limit,
                'alerts': rows,
            }, ensure_ascii=False, indent=2, default=str))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"告警 · 共 {total} 条，第 {page}/{total_pages} 页（本页 {len(rows)} 条）"
            ))
            for r in rows:
                self.stdout.write(
                    f"  id={r.get('id')}\tchannel={r.get('channel')}\t"
                    f"type={r.get('alert_type')}\tscore={r.get('score')}\t"
                    f"created_at={r.get('created_at')}\tllm={r.get('llm_verdict')}\t"
                    f"human={r.get('human_verdict')}\tfinal={r.get('final_status')}"
                )

    def _do_annotate(self, opts):
        c = _ensure_ready()
        ids = _parse_id_list(opts['ids'])
        if not ids:
            raise CommandError("--annotate 需要 --ids（逗号分隔）")
        verdict = opts.get('verdict_value')
        if not verdict:
            raise CommandError("--annotate 需要 --verdict_value（real/false_alarm/uncertain）")
        n = c.sqlite.update_alert_verdict_by_ids(ids, verdict, is_llm=False)
        self.stdout.write(self.style.SUCCESS(
            f"已标注 {n}/{len(ids)} 条为 {verdict}"
        ))

    def _do_delete(self, opts):
        c = _ensure_ready()
        ids = _parse_id_list(opts['ids'])
        if not ids:
            raise CommandError("--delete 需要 --ids（逗号分隔）")
        n = c.sqlite.delete_by_ids('alert_records', ids)
        self.stdout.write(self.style.SUCCESS(
            f"已移到回收站 {n}/{len(ids)} 条"
        ))

    def _do_create(self, opts):
        c = _ensure_ready()
        channel = opts.get('channel')
        if not channel:
            raise CommandError("--create 需要 --channel")
        score = opts.get('score')
        if score is None:
            raise CommandError("--create 需要 --score（数字）")
        # 时间解析
        from phm_site.views_admin import _parse_iso_or_float
        created_at = _parse_iso_or_float(opts.get('ts')) if opts.get('ts') else None
        new_id = c.sqlite.insert_alert_manual(
            channel=channel, score=score, message=opts.get('message', ''),
            created_at=created_at,
        )
        if new_id is None:
            raise CommandError("插入失败")
        self.stdout.write(self.style.SUCCESS(
            f"已补录告警 id={new_id} (channel={channel}, score={score})"
        ))

    def _do_export(self, opts):
        import csv
        import io
        import datetime as _dt
        c = _ensure_ready()
        fmt = opts['output_format'] if opts['output_format'] != 'text' else 'csv'

        # 优先 ids，否则按筛选
        if opts.get('ids'):
            ids = _parse_id_list(opts['ids'])
            rows = []
            for aid in ids:
                try:
                    row = c.sqlite.get_alert_by_id(aid)
                except Exception:
                    row = None
                if row:
                    rows.append(row)
        else:
            filters = self._build_filters(opts)
            rows = c.sqlite.query_alerts_filtered(
                channel=filters['channel'], alert_type=filters['alert_type'],
                status=filters['status'], verdict=filters['verdict'],
                start_ts=filters['start_ts'], end_ts=filters['end_ts'],
                limit=100000,
            )

        def _iso(ts):
            if not ts:
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

        out_stream = open(opts['out'], 'w', encoding='utf-8', newline='') if opts['out'] else self.stdout

        if fmt == 'json':
            payload = _json.dumps({'count': len(serialised), 'alerts': serialised},
                                  ensure_ascii=False, indent=2)
            if opts['out']:
                out_stream.write(payload)
                out_stream.close()
                self.stdout.write(self.style.SUCCESS(
                    f"已导出 {len(serialised)} 条到 {opts['out']}"
                ))
            else:
                self.stdout.write(payload)
        else:  # csv
            # 写文件时加 BOM，stdout 不加（避免终端显示问题）
            if opts['out']:
                out_stream.write('\ufeff')  # BOM
            writer = csv.writer(out_stream)
            writer.writerow(['channel', 'timestamp', 'raw_value', 'anomaly_score',
                             'received_at_iso'])
            for row in serialised:
                writer.writerow([row['channel'], row['timestamp'], row['raw_value'],
                                 row['anomaly_score'], row['received_at_iso']])
            if opts['out']:
                out_stream.close()
                self.stdout.write(self.style.SUCCESS(
                    f"已导出 {len(serialised)} 条到 {opts['out']}"
                ))
