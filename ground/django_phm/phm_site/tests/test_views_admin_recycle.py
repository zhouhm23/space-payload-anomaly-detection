"""第 3 页：回收站测试。

覆盖：
  (a) Helper 纯函数：_parse_recycle_table / _parse_recycle_limit / _parse_id_list
      / _verdict_badge / _alert_type_badge
  (b) 视图访问：匿名跳登录 / staff 可读 / 超管可改 / Container 未就绪占位页
  (c) AJAX 端点权限：restore/purge 非 staff 403、staff 403、超管 200
  (d) AJAX 业务逻辑：成功 / 空 ids / 非法 table / 服务未就绪

测试用 Django TestCase + force_login。Container 与 SQLiteStore 走 mock。
"""
from __future__ import annotations

import json
from unittest import mock

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm_site.views_admin import (
    _alert_type_badge,
    _parse_id_list,
    _parse_recycle_limit,
    _parse_recycle_table,
    _verdict_badge,
)


# ════════════════════════════════════════════════════════════════════════════
# Helper 纯函数测试（不依赖 Django DB）
# ════════════════════════════════════════════════════════════════════════════

class ParseRecycleTableTest(TestCase):

    def test_valid_keys(self):
        for k, expected_sql in [('alerts', 'alert_records'),
                                 ('detections', 'detection_results'),
                                 ('diagnoses', 'diagnosis_records')]:
            key, sql, label = _parse_recycle_table(k)
            self.assertEqual(key, k)
            self.assertEqual(sql, expected_sql)
            self.assertTrue(label)

    def test_invalid_falls_back_to_alerts(self):
        key, sql, _ = _parse_recycle_table('unknown')
        self.assertEqual(key, 'alerts')
        self.assertEqual(sql, 'alert_records')

    def test_none_falls_back(self):
        key, _, _ = _parse_recycle_table(None)
        self.assertEqual(key, 'alerts')

    def test_empty_falls_back(self):
        key, _, _ = _parse_recycle_table('')
        self.assertEqual(key, 'alerts')


class ParseRecycleLimitTest(TestCase):

    def test_default(self):
        self.assertEqual(_parse_recycle_limit(None), 200)
        self.assertEqual(_parse_recycle_limit(''), 200)

    def test_valid_int_string(self):
        self.assertEqual(_parse_recycle_limit('50'), 50)

    def test_clamp_to_max(self):
        self.assertEqual(_parse_recycle_limit('5000'), 1000)

    def test_clamp_to_min(self):
        self.assertEqual(_parse_recycle_limit('0'), 1)
        self.assertEqual(_parse_recycle_limit('-5'), 1)

    def test_invalid_falls_back(self):
        self.assertEqual(_parse_recycle_limit('abc'), 200)


class ParseIdListTest(TestCase):

    def test_list_input(self):
        self.assertEqual(_parse_id_list([1, 2, 3]), [1, 2, 3])

    def test_list_with_invalid_entries(self):
        self.assertEqual(_parse_id_list([1, 'a', None, -5, 0, 2]), [1, 2])

    def test_csv_string(self):
        self.assertEqual(_parse_id_list("1,2,3"), [1, 2, 3])
        self.assertEqual(_parse_id_list("1, 2 ,3"), [1, 2, 3])

    def test_csv_with_garbage(self):
        self.assertEqual(_parse_id_list("1,abc,2,,3"), [1, 2, 3])

    def test_single_int(self):
        self.assertEqual(_parse_id_list(42), [42])

    def test_rejects_zero_and_negative(self):
        self.assertEqual(_parse_id_list([0, -1, -100]), [])

    def test_none_returns_empty(self):
        self.assertEqual(_parse_id_list(None), [])

    def test_empty_string(self):
        self.assertEqual(_parse_id_list(''), [])

    def test_unsupported_type(self):
        self.assertEqual(_parse_id_list({'a': 1}), [])
        self.assertEqual(_parse_id_list(3.14), [])


class VerdictBadgeTest(TestCase):

    def test_known_verdicts(self):
        self.assertEqual(_verdict_badge('real'), 'phm-badge-red')
        self.assertEqual(_verdict_badge('false_alarm'), 'phm-badge-green')
        self.assertEqual(_verdict_badge('uncertain'), 'phm-badge-yellow')

    def test_unknown_falls_back_gray(self):
        self.assertEqual(_verdict_badge(None), 'phm-badge-gray')
        self.assertEqual(_verdict_badge('bogus'), 'phm-badge-gray')


class AlertTypeBadgeTest(TestCase):

    def test_known_types(self):
        self.assertEqual(_alert_type_badge('measured'), 'phm-badge-red')
        self.assertEqual(_alert_type_badge('predicted'), 'phm-badge-yellow')
        self.assertEqual(_alert_type_badge('joint'), 'phm-badge-cyan')

    def test_unknown(self):
        self.assertEqual(_alert_type_badge(None), 'phm-badge-gray')


class FinalStatusBadgeTest(TestCase):
    """_final_status_badge 综合状态徽章映射。"""

    def test_verdict_values(self):
        from phm_site.views_admin import _final_status_badge
        self.assertEqual(_final_status_badge('real'), 'phm-badge-red')
        self.assertEqual(_final_status_badge('false_alarm'), 'phm-badge-green')
        self.assertEqual(_final_status_badge('uncertain'), 'phm-badge-yellow')

    def test_status_values(self):
        from phm_site.views_admin import _final_status_badge
        self.assertEqual(_final_status_badge('confirmed'), 'phm-badge-blue')
        self.assertEqual(_final_status_badge('false'), 'phm-badge-green')
        self.assertEqual(_final_status_badge('pending'), 'phm-badge-yellow')
        self.assertEqual(_final_status_badge('active'), 'phm-badge-gray')

    def test_unknown_falls_back_gray(self):
        from phm_site.views_admin import _final_status_badge
        self.assertEqual(_final_status_badge(None), 'phm-badge-gray')
        self.assertEqual(_final_status_badge('bogus'), 'phm-badge-gray')


class BuildSensorMetaTest(TestCase):
    """_build_sensor_meta 从 device_tree 构造 channel → 传感器名 + 单位 映射。"""

    def test_extracts_sensors(self):
        from phm_site.views_admin import _build_sensor_meta
        cfg = mock.Mock()
        cfg.load.return_value = {
            'device_tree': [
                {'type': 'folder', 'name': 'f1', 'children': [
                    {'type': 'sensor', 'name': '传感器 C-1',
                     'channelName': 'C-1', 'unit': 'A'},
                    {'type': 'sensor', 'name': 'D-14 传感器',
                     'channelName': 'D-14', 'unit': 'mm'},
                ]},
            ]
        }
        meta = _build_sensor_meta(cfg)
        self.assertEqual(meta['C-1']['sensor_name'], '传感器 C-1')
        self.assertEqual(meta['C-1']['unit'], 'A')
        self.assertEqual(meta['D-14']['sensor_name'], 'D-14 传感器')

    def test_sensor_without_channel_falls_back_to_name(self):
        from phm_site.views_admin import _build_sensor_meta
        cfg = mock.Mock()
        cfg.load.return_value = {
            'device_tree': [
                {'type': 'sensor', 'name': 'only-name', 'unit': 'V'},
            ]
        }
        meta = _build_sensor_meta(cfg)
        self.assertIn('only-name', meta)

    def test_none_config_returns_empty(self):
        from phm_site.views_admin import _build_sensor_meta
        self.assertEqual(_build_sensor_meta(None), {})

    def test_load_exception_returns_empty(self):
        from phm_site.views_admin import _build_sensor_meta
        cfg = mock.Mock()
        cfg.load.side_effect = RuntimeError("io error")
        self.assertEqual(_build_sensor_meta(cfg), {})


# ════════════════════════════════════════════════════════════════════════════
# 视图访问测试（RecycleView）
# ════════════════════════════════════════════════════════════════════════════

class RecycleViewAccessTest(TestCase):
    """GET /admin/phm_site/recycle/ 访问权限与渲染。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_recycle')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def _mock_container(self, rows, with_config=True):
        """构造一个 mock Container，sqlite.query_deleted 返回 rows。

        with_config=True 时给 c.config 一个最小 mock（用于 sensor_meta 反查）。
        """
        c = mock.Mock()
        c.sqlite.query_deleted.return_value = rows
        if with_config:
            c.config.load.return_value = {
                'device_tree': [
                    {'type': 'sensor', 'name': '传感器 C-1',
                     'channelName': 'C-1', 'unit': 'A (归一化)'},
                ]
            }
        return c

    def test_anonymous_redirects_to_login(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/admin/login/', resp['Location'])

    def test_staff_can_read_empty(self):
        """staff 可访问，看到空回收站。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container([])
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '回收站为空')
        # staff 看不到批量按钮（is_superuser=False）——
        # 注意：JS 代码块里仍含字符串 'phm-recycle-restore-btn'（不可达分支），
        # 但 DOM 里没有 <button id="phm-recycle-restore-btn">。这里检查按钮元素。
        self.assertNotContains(resp, '<button class="phm-btn phm-btn-primary phm-btn-sm" id="phm-recycle-restore-btn"')
        self.assertNotContains(resp, '<button class="phm-btn phm-btn-danger phm-btn-sm" id="phm-recycle-purge-btn"')

    def test_staff_sees_readonly_warning(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container([])
            resp = self.client.get(self.url)
        self.assertContains(resp, '当前角色为')

    def test_superuser_sees_buttons(self):
        """超管 + 有数据时看到顶部批量工具栏。"""
        self.client.force_login(self.superuser)
        rows = [{
            'id': 1, 'channel': 'C-1', 'alert_type': 'measured', 'score': 0.5,
            'created_at': 1719000000.0, 'status': 'active',
            'llm_verdict': None, 'human_verdict': None,
            'raw_value': 0.123, 'deleted_at': 1719100000.0,
            'final_status': 'active',
        }]
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container(rows)
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'phm-recycle-restore-btn')
        self.assertContains(resp, 'phm-recycle-purge-btn')

    def test_superuser_sees_no_toolbar_when_empty(self):
        """超管 + 空回收站时不渲染工具栏（无数据可批量操作）。

        注意：JS 代码块里仍含 'phm-recycle-restore-btn' 字符串引用，
        但 DOM 里没有 <button id="phm-recycle-restore-btn">。这里检查按钮元素。
        """
        self.client.force_login(self.superuser)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container([])
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, '<button class="phm-btn phm-btn-primary phm-btn-sm" id="phm-recycle-restore-btn"')
        self.assertNotContains(resp, 'phm-recycle-toolbar')

    def test_renders_alert_columns(self):
        """alert 行应渲染需求书要求的全部列（除操作列）。"""
        self.client.force_login(self.superuser)
        rows = [{
            'id': 12, 'channel': 'C-1', 'alert_type': 'measured',
            'score': 0.7, 'created_at': 1719000000.0,
            'status': 'active', 'verified_at': None,
            'llm_verdict': 'real', 'human_verdict': 'false_alarm',
            'raw_snapshot': [0.1, 0.2, 0.3], 'raw_value': 0.3,
            'deleted_at': 1719100000.0, 'final_status': 'false_alarm',
        }]
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container(rows)
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # 传感器名（来自 device_tree mock）
        self.assertContains(resp, '传感器 C-1')
        # 遥测值（raw_snapshot 末点）+ 单位
        self.assertContains(resp, '0.300')
        self.assertContains(resp, 'A (归一化)')
        # LLM 状态 / 人工状态 / 综合状态 都应作为独立徽章渲染
        self.assertContains(resp, '>real<')
        self.assertContains(resp, '>false_alarm<')
        # 类型徽章
        self.assertContains(resp, '>measured<')

    def test_renders_unknown_channel_falls_back(self):
        """告警 channel 不在 device_tree 里时，传感器名回退为 channel 名。"""
        self.client.force_login(self.superuser)
        rows = [{
            'id': 99, 'channel': 'UNKNOWN-CH', 'alert_type': 'measured',
            'score': 0.5, 'created_at': 1719000000.0, 'status': 'active',
            'llm_verdict': None, 'human_verdict': None, 'raw_value': None,
            'deleted_at': 1719100000.0, 'final_status': 'active',
        }]
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container(rows)
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # 传感器名回退为 channel
        self.assertContains(resp, 'UNKNOWN-CH')

    def test_table_tab_switch(self):
        """?table=detections 应让该 tab active。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container([])
            resp = self.client.get(self.url, {'table': 'detections'})
        self.assertEqual(resp.status_code, 200)
        # query_deleted 应被调用时传 detection_results
        sb.get_container.return_value.sqlite.query_deleted.assert_called_once_with(
            'detection_results', limit=200
        )

    def test_invalid_table_falls_back(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container([])
            self.client.get(self.url, {'table': 'garbage'})
        sb.get_container.return_value.sqlite.query_deleted.assert_called_once_with(
            'alert_records', limit=200
        )

    def test_container_not_ready_renders_state_page(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'initializing'
            sb.get_init_error.return_value = None
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '正在初始化')


# ════════════════════════════════════════════════════════════════════════════
# AJAX 端点权限测试
# ════════════════════════════════════════════════════════════════════════════

class RecycleAjaxPermissionTest(TestCase):
    """restore / purge AJAX 端点权限校验。"""

    def setUp(self):
        self.client = Client()
        self.restore_url = reverse('phm_admin_recycle_restore')
        self.purge_url = reverse('phm_admin_recycle_purge')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def _payload(self, table='alerts', ids=None):
        return json.dumps({'table': table, 'ids': ids or [1, 2]})

    def test_anonymous_restore_redirects_to_login(self):
        """匿名 → @staff_member_required 装饰器返回 302。"""
        resp = self.client.post(self.restore_url, self._payload(),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/admin/login/', resp['Location'])

    def test_staff_restore_forbidden(self):
        """staff 登录但非超管 → _require_superuser 返回 403。"""
        self.client.force_login(self.staff)
        resp = self.client.post(self.restore_url, self._payload(),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 403)
        body = resp.json()
        self.assertEqual(body['status'], 'error')

    def test_staff_purge_forbidden(self):
        self.client.force_login(self.staff)
        resp = self.client.post(self.purge_url, self._payload(),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 403)

    def test_superuser_restore_passes_superuser_check(self):
        """超管 → _require_superuser 通过，进入业务逻辑。"""
        self.client.force_login(self.superuser)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                sqlite=mock.Mock(restore=mock.Mock(return_value=2))
            )
            resp = self.client.post(self.restore_url, self._payload(),
                                    content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['restored'], 2)


# ════════════════════════════════════════════════════════════════════════════
# AJAX 业务逻辑测试
# ════════════════════════════════════════════════════════════════════════════

class RecycleAjaxLogicTest(TestCase):
    """restore / purge 业务逻辑（超管身份下）。"""

    def setUp(self):
        self.client = Client()
        self.restore_url = reverse('phm_admin_recycle_restore')
        self.purge_url = reverse('phm_admin_recycle_purge')
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )
        self.client.force_login(self.superuser)

    def _mock_sb(self, sql_table, fn_name, return_value):
        sb = mock.MagicMock()
        sb.get_state.return_value = 'ready'
        sqlite = mock.Mock()
        getattr(sqlite, fn_name).return_value = return_value
        sb.get_container.return_value = mock.Mock(sqlite=sqlite)
        return sb

    # ── restore 成功 ─────────────────────────────────────────────

    def test_restore_alerts_success(self):
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                sqlite=mock.Mock(restore=mock.Mock(return_value=2))
            )
            resp = self.client.post(
                self.restore_url,
                json.dumps({'table': 'alerts', 'ids': [10, 11]}),
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['restored'], 2)
        # 验证调用了 SQLiteStore.restore('alert_records', [10, 11])
        sb.get_container.return_value.sqlite.restore.assert_called_once_with(
            'alert_records', [10, 11]
        )

    def test_restore_detections_table(self):
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                sqlite=mock.Mock(restore=mock.Mock(return_value=1))
            )
            resp = self.client.post(
                self.restore_url,
                json.dumps({'table': 'detections', 'ids': [5]}),
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        sb.get_container.return_value.sqlite.restore.assert_called_once_with(
            'detection_results', [5]
        )

    # ── restore 失败/边界 ────────────────────────────────────────

    def test_restore_empty_ids_returns_400(self):
        resp = self.client.post(
            self.restore_url,
            json.dumps({'table': 'alerts', 'ids': []}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()['status'], 'error')

    def test_restore_garbage_ids_returns_400(self):
        resp = self.client.post(
            self.restore_url,
            json.dumps({'table': 'alerts', 'ids': ['x', 0, -1]}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_restore_invalid_json_returns_400(self):
        resp = self.client.post(
            self.restore_url, 'not-a-json',
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_restore_invalid_table_falls_back_to_alerts(self):
        """非法 table 兜底为 alerts，仍正常调 SQLiteStore.restore。"""
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                sqlite=mock.Mock(restore=mock.Mock(return_value=0))
            )
            resp = self.client.post(
                self.restore_url,
                json.dumps({'table': 'unknown', 'ids': [1]}),
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        sb.get_container.return_value.sqlite.restore.assert_called_once_with(
            'alert_records', [1]
        )

    def test_restore_container_not_ready_returns_503(self):
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'initializing'
            sb.get_init_error.return_value = None
            resp = self.client.post(
                self.restore_url,
                json.dumps({'table': 'alerts', 'ids': [1]}),
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 503)

    # ── purge 成功 ───────────────────────────────────────────────

    def test_purge_success(self):
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                sqlite=mock.Mock(purge_by_ids=mock.Mock(return_value=3))
            )
            resp = self.client.post(
                self.purge_url,
                json.dumps({'table': 'alerts', 'ids': [1, 2, 3]}),
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['purged'], 3)
        sb.get_container.return_value.sqlite.purge_by_ids.assert_called_once_with(
            'alert_records', [1, 2, 3]
        )

    def test_purge_diagnoses_table(self):
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                sqlite=mock.Mock(purge_by_ids=mock.Mock(return_value=1))
            )
            resp = self.client.post(
                self.purge_url,
                json.dumps({'table': 'diagnoses', 'ids': [7]}),
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        sb.get_container.return_value.sqlite.purge_by_ids.assert_called_once_with(
            'diagnosis_records', [7]
        )

    def test_purge_empty_ids_returns_400(self):
        resp = self.client.post(
            self.purge_url,
            json.dumps({'table': 'alerts', 'ids': []}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_purge_service_exception_returns_500(self):
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                sqlite=mock.Mock(purge_by_ids=mock.Mock(side_effect=RuntimeError("db lock")))
            )
            resp = self.client.post(
                self.purge_url,
                json.dumps({'table': 'alerts', 'ids': [1]}),
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.json()['status'], 'error')

    # ── HTTP 方法限制 ────────────────────────────────────────────

    def test_get_restore_not_allowed(self):
        """restore 端点只接受 POST。"""
        resp = self.client.get(self.restore_url)
        self.assertEqual(resp.status_code, 405)

    def test_get_purge_not_allowed(self):
        resp = self.client.get(self.purge_url)
        self.assertEqual(resp.status_code, 405)
