"""System / theme config management command.

Agent 友好双通道（AGENTS.md §结构化设计规范 第 5 条）：与「系统设置」
HTTP API 并行的 CLI 通道，直调 service 层，不重复业务逻辑。

子命令：
  --get  --category system|theme  --key thresholds.anomaly [--format json|text]
  --set  --category system|theme  --key thresholds.anomaly --value 0.42
  --list --category system|theme|calibration [--format json|text]

--key 用点号分隔 section.key（与 settings 页面的「变量名」列一致）。

注意：calibration 只支持 --list / --get，不支持 --set（离线标定产物）。

示例：
  python manage.py phm_config --list --category system --format json
  python manage.py phm_config --get --category theme --key colors.blue
  python manage.py phm_config --set --category system --key thresholds.anomaly --value 0.42
"""
from __future__ import annotations

import json as _json

from django.core.management.base import BaseCommand, CommandError

from phm.services.system_config_service import get_system_config
from phm.services.theme_service import get_theme
from phm_site.views_admin import (
    _CALIBRATION_PATH, _SETTINGS_CATEGORY_KEYS, _parse_settings_category,
)


def _parse_key(key: str) -> tuple[str, str]:
    """把 'section.key' 切成 (section, key)。无点号 → CommandError。"""
    if not key or "." not in key:
        raise CommandError(f"--key 必须形如 'section.key'，实际：{key!r}")
    section, _, sub = key.partition(".")
    if not section or not sub:
        raise CommandError(f"--key 格式非法：{key!r}")
    return section, sub


def _value_coerce(raw: str | None) -> object:
    """把命令行字符串智能转成 bool/int/float/str。

    service 层会再做一次类型校验，这里只是尽量让 --value 0.42 被识别为
    float、--value true 被识别为 bool。
    """
    if raw is None:
        raise CommandError("--set 需要 --value")
    low = raw.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    # int？
    try:
        return int(raw)
    except ValueError:
        pass
    # float？
    try:
        return float(raw)
    except ValueError:
        pass
    # JSON 对象/数组？
    try:
        return _json.loads(raw)
    except (ValueError, TypeError):
        pass
    return raw


class Command(BaseCommand):
    help = "系统设置：列出/读取/修改 system_config.json 与 ui_theme.json"

    def add_arguments(self, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--list', action='store_true', help='列出全部配置项')
        group.add_argument('--get', action='store_true', help='读取单个值')
        group.add_argument('--set', action='store_true', help='修改单个值')

        parser.add_argument(
            '--category', default='system',
            choices=sorted(_SETTINGS_CATEGORY_KEYS),
            help='配置类：system / theme / calibration（默认 system）',
        )
        parser.add_argument(
            '--key', default='',
            help='section.key 形式的键名（--get / --set 必填）',
        )
        parser.add_argument(
            '--value', default=None,
            help='新值（--set 必填；自动识别 bool/int/float/JSON/str）',
        )
        parser.add_argument(
            '--format', dest='output_format', default='text',
            choices=['text', 'json'],
            help='输出格式（默认 text）',
        )

    def handle(self, *args, **opts):
        category = _parse_settings_category(opts['category'])

        if opts['list']:
            self._do_list(category, opts)
            return
        if opts['get']:
            self._do_get(category, opts)
            return
        if opts['set']:
            self._do_set(category, opts)
            return

    # ── 子操作 ──────────────────────────────────────────────────

    def _do_list(self, category, opts):
        if category == 'system':
            svc = get_system_config()
            raw = svc.raw_with_docs()
            names = svc.display_names()
        elif category == 'theme':
            svc = get_theme()
            raw = svc.raw_with_docs()
            names = svc.display_names()
        else:  # calibration
            self._list_calibration(opts)
            return

        # 扁平化为 [{section, key, name, value, doc}]
        flat = []
        for section, sec_values in raw.items():
            if section.startswith('_') or not isinstance(sec_values, dict):
                continue
            sec_names = names.get(section, {})
            for key, value in sec_values.items():
                if key.startswith('_'):
                    continue
                flat.append({
                    'section': section,
                    'key': key,
                    'name': sec_names.get(key, key),
                    'value': value,
                    'section_doc': sec_values.get('_doc', ''),
                })

        if opts['output_format'] == 'json':
            self.stdout.write(_json.dumps({
                'category': category, 'count': len(flat), 'items': flat,
            }, ensure_ascii=False, indent=2))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"{category} 配置 · 共 {len(flat)} 项"
            ))
            for it in flat:
                self.stdout.write(
                    f"  {it['section']}.{it['key']}\t= {it['value']!r}\t# {it['name']}"
                )

    def _list_calibration(self, opts):
        import os
        if not os.path.exists(_CALIBRATION_PATH):
            if opts['output_format'] == 'json':
                self.stdout.write(_json.dumps({
                    'category': 'calibration', 'count': 0, 'items': [],
                    'note': 'channel_calibration.json 不存在',
                }))
            else:
                self.stdout.write(self.style.WARNING(
                    "channel_calibration.json 不存在（系统未标定）"
                ))
            return
        try:
            with open(_CALIBRATION_PATH, encoding='utf-8') as f:
                raw = _json.load(f)
        except Exception as e:
            raise CommandError(f"读取 calibration 失败：{e}")
        flat = []
        for ch, cfg in raw.items():
            if ch.startswith('_') or not isinstance(cfg, dict):
                continue
            for k, v in cfg.items():
                if k.startswith('_'):
                    continue
                if isinstance(v, list):
                    v_repr = f"<{len(v)} 项>"
                else:
                    v_repr = v
                flat.append({'channel': ch, 'key': k, 'value': v_repr})
        if opts['output_format'] == 'json':
            self.stdout.write(_json.dumps({
                'category': 'calibration', 'count': len(flat), 'items': flat,
            }, ensure_ascii=False, indent=2))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"calibration · 共 {len(flat)} 项（只读）"
            ))
            for it in flat:
                self.stdout.write(
                    f"  {it['channel']}.{it['key']}\t= {it['value']!r}"
                )

    def _do_get(self, category, opts):
        if category == 'calibration':
            raise CommandError("calibration 暂不支持 --get（请用 --list）")
        section, key = _parse_key(opts['key'])
        if category == 'system':
            value = get_system_config().get(section, key)
        else:
            value = get_theme().as_dict().get(section, {}).get(key)
        if value is None:
            raise CommandError(f"{section}.{key} 不存在")
        if opts['output_format'] == 'json':
            self.stdout.write(_json.dumps({
                'category': category, 'section': section, 'key': key,
                'value': value,
            }, ensure_ascii=False))
        else:
            self.stdout.write(f"{section}.{key} = {value!r}")

    def _do_set(self, category, opts):
        if category == 'calibration':
            raise CommandError("calibration 只读，不支持 --set（离线标定产物）")
        section, key = _parse_key(opts['key'])
        value = _value_coerce(opts['value'])
        if category == 'system':
            result = get_system_config().save(section, key, value)
        else:
            result = get_theme().save(section, key, value)
        if result.get('status') != 'ok':
            raise CommandError(result.get('message', '保存失败'))
        if opts['output_format'] == 'json':
            self.stdout.write(_json.dumps(result, ensure_ascii=False))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"已更新 {category}:{section}.{key}: {result['old']!r} → {result['new']!r}"
            ))
