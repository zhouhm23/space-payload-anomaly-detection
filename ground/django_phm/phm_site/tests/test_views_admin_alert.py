"""第 4 页：告警和预警管理测试。

覆盖：
  (a) Helper 纯函数：_parse_alert_filters / _parse_iso_or_float / _parse_alert_limit
  (b) 视图访问：匿名跳登录 / staff 可读 / 超管可改 / Container 未就绪占位页
  (c) AJAX 端点：detail / annotate / delete / diagnose / diagnose_status / export / create
      覆盖权限（403/302）+ 业务逻辑（成功/空 ids/类型校验/服务未就绪）
"""
from __future__ import annotations

import json
from unittest import mock

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm_site.views_admin import (
    _parse_alert_filters,
    _parse_alert_limit,
    _parse_alert_page,
    _build_page_range,
    _parse_iso_or_float,
)


# ════════════════════════════════════════════════════════════════════════════
# Helper 纯函数测试
# ════════════════════════════════════════════════════════════════════════════

class ParseAlertFiltersTest(TestCase):

    def _build_get(self, **kwargs):
        return kwargs

    def test_empty_returns_all_none(self):
        f = _parse_alert_filters({})
        self.assertIsNone(f['channel'])
        self.assertIsNone(f['alert_type'])
        self.assertIsNone(f['llm_verdict'])
        self.assertIsNone(f['human_verdict'])
        self.assertIsNone(f['verdict'])
        self.assertIsNone(f['start_ts'])
        self.assertIsNone(f['end_ts'])

    def test_channel_strips_and_clips(self):
        f = _parse_alert_filters({'channel': '  C-1  '})
        self.assertEqual(f['channel'], 'C-1')
        # 超长 channel 截断
        long = 'x' * 100
        f = _parse_alert_filters({'channel': long})
        self.assertEqual(len(f['channel']), 64)

    def test_alert_type_whitelist(self):
        f = _parse_alert_filters({'alert_type': 'measured'})
        self.assertEqual(f['alert_type'], 'measured')
        f = _parse_alert_filters({'alert_type': 'bogus'})
        self.assertIsNone(f['alert_type'])

    def test_llm_verdict_whitelist(self):
        for v in ('real', 'false_alarm', 'uncertain', 'none'):
            f = _parse_alert_filters({'llm_verdict': v})
            self.assertEqual(f['llm_verdict'], v)
        f = _parse_alert_filters({'llm_verdict': 'bogus'})
        self.assertIsNone(f['llm_verdict'])

    def test_human_verdict_whitelist(self):
        for v in ('real', 'false_alarm', 'uncertain', 'none'):
            f = _parse_alert_filters({'human_verdict': v})
            self.assertEqual(f['human_verdict'], v)
        f = _parse_alert_filters({'human_verdict': 'bogus'})
        self.assertIsNone(f['human_verdict'])

    def test_verdict_whitelist(self):
        for v in ('real', 'false_alarm', 'uncertain'):
            f = _parse_alert_filters({'verdict': v})
            self.assertEqual(f['verdict'], v)
        f = _parse_alert_filters({'verdict': 'bogus'})
        self.assertIsNone(f['verdict'])


class ParseIsoOrFloatTest(TestCase):

    def test_unix_seconds(self):
        self.assertEqual(_parse_iso_or_float('1700000000'), 1700000000.0)
        self.assertEqual(_parse_iso_or_float(1700000000.5), 1700000000.5)

    def test_iso_datetime(self):
        ts = _parse_iso_or_float('2026-07-21T12:00:00')
        self.assertIsInstance(ts, float)
        self.assertGreater(ts, 0)

    def test_iso_with_space(self):
        ts = _parse_iso_or_float('2026-07-21 12:00:00')
        self.assertIsInstance(ts, float)

    def test_iso_date_only(self):
        ts = _parse_iso_or_float('2026-07-21')
        self.assertIsInstance(ts, float)

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_iso_or_float('bogus'))
        self.assertIsNone(_parse_iso_or_float(None))
        self.assertIsNone(_parse_iso_or_float(''))


class ParseAlertLimitTest(TestCase):

    def test_default(self):
        self.assertEqual(_parse_alert_limit(None), 20)
        self.assertEqual(_parse_alert_limit(''), 20)

    def test_valid_int(self):
        self.assertEqual(_parse_alert_limit('100'), 100)

    def test_clamp_max(self):
        self.assertEqual(_parse_alert_limit('5000'), 1000)

    def test_clamp_min(self):
        self.assertEqual(_parse_alert_limit('0'), 1)
        self.assertEqual(_parse_alert_limit('-5'), 1)

    def test_invalid_falls_back(self):
        self.assertEqual(_parse_alert_limit('abc'), 20)


class ParseAlertPageTest(TestCase):

    def test_default(self):
        self.assertEqual(_parse_alert_page(None), 1)
        self.assertEqual(_parse_alert_page(''), 1)

    def test_valid_int(self):
        self.assertEqual(_parse_alert_page('3'), 3)

    def test_clamp_min(self):
        self.assertEqual(_parse_alert_page('0'), 1)
        self.assertEqual(_parse_alert_page('-5'), 1)

    def test_clamp_max(self):
        self.assertEqual(_parse_alert_page('999999'), 100000)

    def test_invalid_falls_back(self):
        self.assertEqual(_parse_alert_page('abc'), 1)


class BuildPageRangeTest(TestCase):

    def test_empty_when_zero_pages(self):
        self.assertEqual(_build_page_range(1, 0), [])

    def test_all_pages_when_small(self):
        """total_pages ≤ 7 时返回连续 1..N。"""
        self.assertEqual(_build_page_range(1, 1), [1])
        self.assertEqual(_build_page_range(3, 5), [1, 2, 3, 4, 5])
        self.assertEqual(_build_page_range(4, 7), [1, 2, 3, 4, 5, 6, 7])

    def test_ellipsis_for_large(self):
        """total_pages > 7 时中间用省略号。"""
        # page=1, total=10：[1, 2, 3, '..', 10]
        r = _build_page_range(1, 10)
        self.assertIn(1, r)
        self.assertIn(10, r)
        self.assertIn('..', r)

    def test_current_page_in_middle(self):
        """当前页在中间时，前后各 window 页都显示。"""
        r = _build_page_range(5, 10)
        # 期望 [1, '..', 3, 4, 5, 6, 7, '..', 10]
        self.assertIn(5, r)
        self.assertIn(3, r)
        self.assertIn(7, r)
        self.assertIn(1, r)
        self.assertIn(10, r)

    def test_no_consecutive_ellipsis(self):
        """不出现连续两个省略号。"""
        for page in range(1, 20):
            r = _build_page_range(page, 19)
            for i in range(len(r) - 1):
                self.assertFalse(r[i] == '..' and r[i + 1] == '..',
                                 f'consecutive ellipsis at page={page}: {r}')


# ════════════════════════════════════════════════════════════════════════════
# 视图访问测试
# ════════════════════════════════════════════════════════════════════════════

def _make_mock_container(rows=None, alert_by_id=None, diag_result=None):
    """构造一个 mock Container，sqlite / config 都配齐。"""
    c = mock.Mock()
    c.sqlite.query_alerts_filtered.return_value = rows or []
    # count_alerts_filtered 默认返回 0（与 rows=[] 一致），分页测试可单独覆盖
    c.sqlite.count_alerts_filtered.return_value = len(rows or [])
    c.sqlite.get_alert_by_id.return_value = alert_by_id
    c.sqlite.get_diagnosis.return_value = diag_result
    c.sqlite.update_alert_verdict_by_ids.return_value = 1
    c.sqlite.delete_by_ids.return_value = 1
    c.sqlite.insert_alert_manual.return_value = 42
    c.config.load.return_value = {'device_tree': [
        {'type': 'sensor', 'name': '传感器 C-1', 'channelName': 'C-1', 'unit': 'A'},
    ]}
    return c


def _patch_container(c):
    """patch services_bridge.get_container / get_state，让三态机走 ready 分支。"""
    return (
        mock.patch('phm_site.services_bridge.get_container', return_value=c),
        mock.patch('phm_site.services_bridge.get_state', return_value='ready'),
    )


class AlertViewAccessTest(TestCase):
    """GET /admin/phm_site/alert/ 访问权限与渲染。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_alert')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_anonymous_redirects_to_login(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/admin/login/', resp.url)

    def test_staff_can_access(self):
        self.client.force_login(self.staff)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '告警和预警管理')

    def test_superuser_can_access(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_container_not_ready_renders_state_page(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.services_bridge.get_state', return_value='initializing'):
            resp = self.client.get(self.url)
        # _state.html 应包含"初始化中"提示
        self.assertEqual(resp.status_code, 200)

    def test_filter_params_forwarded_to_query(self):
        """channel/type/llm_verdict 等筛选参数应被传给 sqlite.query_alerts_filtered。"""
        self.client.force_login(self.staff)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            self.client.get(self.url + '?channel=C-1&verdict=real&llm_verdict=none&limit=10')
        # 验证调用参数
        args, kwargs = c.sqlite.query_alerts_filtered.call_args
        self.assertEqual(kwargs.get('channel'), 'C-1')
        self.assertEqual(kwargs.get('verdict'), 'real')
        self.assertEqual(kwargs.get('llm_verdict'), 'none')
        self.assertEqual(kwargs.get('limit'), 10)

    def test_offset_passed_to_query(self):
        """?page=2 应算出 offset=limit 传给 query_alerts_filtered。"""
        self.client.force_login(self.staff)
        c = _make_mock_container()
        c.sqlite.count_alerts_filtered.return_value = 100  # 100 条，limit=50 → 2 页
        p1, p2 = _patch_container(c)
        with p1, p2:
            self.client.get(self.url + '?page=2&limit=50')
        args, kwargs = c.sqlite.query_alerts_filtered.call_args
        self.assertEqual(kwargs.get('offset'), 50)

    def test_chinese_labels_in_decorated_rows(self):
        """数据行应带中文 label（避免 SSR 显示英文 measured/real/active）。"""
        self.client.force_login(self.staff)
        rows = [{
            'id': 1, 'channel': 'C-1', 'alert_type': 'measured', 'score': 0.6,
            'created_at': 1700000000, 'status': 'active',
            'llm_verdict': 'real', 'human_verdict': 'false_alarm',
            'final_status': 'real', 'raw_snapshot': [0.1, 0.2],
        }]
        c = _make_mock_container(rows=rows)
        c.sqlite.count_alerts_filtered.return_value = 1
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        item = resp.context['rows'][0]
        self.assertEqual(item['alert_type_label'], '实测告警')
        self.assertEqual(item['llm_verdict_label'], '实警')
        self.assertEqual(item['human_verdict_label'], '虚警')
        self.assertEqual(item['final_status_label'], '实警')
        # 页面渲染应含中文（不含英文 measured）
        self.assertContains(resp, '实测告警')
        self.assertNotContains(resp, '>measured<')

    def test_joint_alert_type_label(self):
        """联合告警 alert_type_label 应为中文「联合告警」。"""
        self.client.force_login(self.staff)
        rows = [{
            'id': 2, 'channel': 'SUB:数据集', 'alert_type': 'joint', 'score': 0.7,
            'created_at': 1700000000, 'status': 'active',
            'llm_verdict': None, 'human_verdict': None,
            'final_status': 'active', 'raw_snapshot': None,
        }]
        c = _make_mock_container(rows=rows)
        c.sqlite.count_alerts_filtered.return_value = 1
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        item = resp.context['rows'][0]
        self.assertEqual(item['alert_type_label'], '联合告警')


class AlertViewPaginationTest(TestCase):
    """alert_view 分页逻辑测试。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_alert')
        self.staff = User.objects.create_user(
            username='staff2', password='pw', is_staff=True
        )

    def _make_rows(self, n):
        """造 n 条 mock alert 行。"""
        return [
            {'id': i, 'channel': 'C-1', 'alert_type': 'measured', 'score': 0.5,
             'created_at': 1700000000 + i, 'status': 'active',
             'llm_verdict': None, 'human_verdict': None,
             'raw_snapshot': None, 'score_snapshot': None,
             'ingested_at': 1700000000 + i, 'final_status': 'active',
             'message': '', 'verified_at': None}
            for i in range(n)
        ]

    def test_default_page_is_1(self):
        """无 ?page= 参数时默认第 1 页。"""
        self.client.force_login(self.staff)
        c = _make_mock_container(self._make_rows(3))
        c.sqlite.count_alerts_filtered.return_value = 3
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['page'], 1)
        self.assertEqual(resp.context['total_pages'], 1)
        self.assertEqual(resp.context['total_count'], 3)

    def test_total_pages_calculation(self):
        """total_pages = ceil(total_count / limit)。"""
        self.client.force_login(self.staff)
        c = _make_mock_container(self._make_rows(50))
        c.sqlite.count_alerts_filtered.return_value = 120  # limit=50 → 3 页
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?limit=50')
        self.assertEqual(resp.context['total_pages'], 3)

    def test_page_beyond_last_clamped(self):
        """page 超出 total_pages 时兜底到最后一页。"""
        self.client.force_login(self.staff)
        c = _make_mock_container(self._make_rows(10))
        c.sqlite.count_alerts_filtered.return_value = 55  # limit=50 → 2 页
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?page=99&limit=50')
        self.assertEqual(resp.context['page'], 2)  # 兜底到最后一页
        # offset 应是 (2-1)*50 = 50
        args, kwargs = c.sqlite.query_alerts_filtered.call_args
        self.assertEqual(kwargs.get('offset'), 50)

    def test_zero_results_total_pages_is_1(self):
        """无数据时 total_pages=1（不显示分页栏）。"""
        self.client.force_login(self.staff)
        c = _make_mock_container([])
        c.sqlite.count_alerts_filtered.return_value = 0
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.context['total_pages'], 1)
        self.assertEqual(resp.context['total_count'], 0)

    def test_pagination_not_rendered_when_one_page(self):
        """只有 1 页时不渲染分页控件。"""
        self.client.force_login(self.staff)
        c = _make_mock_container(self._make_rows(3))
        c.sqlite.count_alerts_filtered.return_value = 3
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertNotContains(resp, 'phm-pagination')

    def test_pagination_rendered_when_multiple_pages(self):
        """多页时渲染分页控件 + 页码。"""
        self.client.force_login(self.staff)
        c = _make_mock_container(self._make_rows(50))
        c.sqlite.count_alerts_filtered.return_value = 150  # 3 页
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?limit=50')
        self.assertContains(resp, 'phm-pagination')
        self.assertContains(resp, '第 1/3 页')

    def test_page_size_options_in_context(self):
        """context 含 page_size_options（供每页数量下拉渲染）。"""
        self.client.force_login(self.staff)
        c = _make_mock_container(self._make_rows(3))
        c.sqlite.count_alerts_filtered.return_value = 3
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        opts = resp.context['page_size_options']
        self.assertEqual(opts, [20, 50, 100, 200])
        # 当前 limit=50 应被选中（HTML 含 selected）
        self.assertContains(resp, 'phm-page-size-select')

    def test_page_size_select_renders_all_options(self):
        """每页数量下拉渲染所有候选值。"""
        self.client.force_login(self.staff)
        c = _make_mock_container(self._make_rows(3))
        c.sqlite.count_alerts_filtered.return_value = 3
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        for n in (20, 50, 100, 200):
            self.assertContains(resp, 'value="{}"'.format(n))

    def test_page_jump_input_renders_when_multiple_pages(self):
        """多页时渲染跳转输入框。"""
        self.client.force_login(self.staff)
        c = _make_mock_container(self._make_rows(50))
        c.sqlite.count_alerts_filtered.return_value = 150
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?limit=50')
        self.assertContains(resp, 'phm-page-jump-input')
        self.assertContains(resp, 'phm-page-jump-btn')
        self.assertContains(resp, 'max="3"')  # total_pages=3


# ════════════════════════════════════════════════════════════════════════════
# AJAX 端点测试
# ════════════════════════════════════════════════════════════════════════════

class AlertDetailApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.url = reverse('phm_admin_alert_detail', args=[1])

    def test_anonymous_redirects(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)

    def test_not_found(self):
        self.client.force_login(self.staff)
        c = _make_mock_container(alert_by_id=None)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 404)

    def test_success(self):
        self.client.force_login(self.staff)
        alert = {
            'id': 1, 'channel': 'C-1', 'alert_type': 'measured',
            'score': 0.8, 'message': 'x', 'created_at': 1700000000,
            'status': 'active', 'llm_verdict': None, 'human_verdict': None,
            'final_status': 'active', 'raw_snapshot': [1, 2, 3],
            'score_snapshot': [0.1, 0.5, 0.8],
        }
        c = _make_mock_container(alert_by_id=alert,
                                  diag_result={'diagnosis': 'real',
                                                'context_summary': {}})
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['alert']['channel'], 'C-1')
        self.assertEqual(body['diagnosis']['text'], 'real')


class AlertAnnotateApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_alert_annotate')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_staff_gets_403(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url, {'ids': [1], 'verdict': 'real'},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 403)

    def test_superuser_success(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'ids': [1, 2], 'verdict': 'real'},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['updated'], 1)

    def test_invalid_verdict_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'ids': [1], 'verdict': 'bogus'},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)

    def test_empty_ids_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'ids': [], 'verdict': 'real'},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)


class AlertDeleteApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_alert_delete')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_staff_gets_403(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url, {'ids': [1]},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 403)

    def test_superuser_success(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'ids': [1]},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['status'], 'ok')

    def test_calls_delete_by_ids_with_alert_records(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            self.client.post(self.url, {'ids': [7, 8]},
                              content_type='application/json')
        c.sqlite.delete_by_ids.assert_called_with('alert_records', [7, 8])


class AlertDiagnoseApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_alert_diagnose')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_staff_can_trigger(self):
        """LLM 诊断是只读语义，staff 可触发（不是 _require_superuser）。"""
        self.client.force_login(self.staff)
        alert = {'id': 1, 'channel': 'C-1', 'alert_type': 'measured',
                 'created_at': 1700000000}
        c = _make_mock_container(alert_by_id=alert)
        c.diagnosis = mock.Mock()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'ids': [1]},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['status'], 'ok')

    def test_empty_ids_returns_400(self):
        self.client.force_login(self.staff)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'ids': []},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)


class AlertDiagnoseOneApiTest(TestCase):
    """单条同步诊断 API（抽屉内「诊断/重新诊断」按钮用）。"""

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def _url(self, aid):
        return reverse('phm_admin_alert_diagnose_one', args=[aid])

    def test_staff_can_trigger_sync_diagnose(self):
        """单条同步诊断：staff 可触发，返回诊断文本与 verdict。"""
        self.client.force_login(self.staff)
        alert = {'id': 5, 'channel': 'C-1', 'alert_type': 'measured',
                 'created_at': 1700000000, 'llm_verdict': 'real', 'final_status': 'real'}
        fresh_alert = dict(alert, llm_verdict='real', final_status='real')
        c = _make_mock_container(alert_by_id=alert)
        c.diagnosis = mock.Mock()
        # diagnose() 返回结果 dict（与 DiagnosisService.diagnose 签名一致）
        c.diagnosis.diagnose.return_value = {
            'diagnosis': '该传感器存在异常漂移',
            'llm_verdict': 'real',
            'error': None,
            'elapsed_sec': 1.23,
            'cached': False,
        }
        # 二次 get_alert_by_id（API 内部重读最新行拿 verdict）
        c.sqlite.get_alert_by_id.side_effect = [alert, fresh_alert]
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self._url(5), {'force_refresh': True},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['id'], 5)
        self.assertEqual(body['diagnosis_text'], '该传感器存在异常漂移')
        self.assertEqual(body['llm_verdict'], 'real')
        self.assertEqual(body['final_status'], 'real')
        self.assertEqual(body['elapsed_sec'], 1.23)
        # 验证 diagnosis service 被正确调用
        c.diagnosis.diagnose.assert_called_once()
        call_args = c.diagnosis.diagnose.call_args
        # channel 是位置参数（views_admin: diag.diagnose(row['channel'], alert_type=..., ...))
        self.assertEqual(call_args.args[0], 'C-1')
        self.assertEqual(call_args.kwargs.get('alert_type'), 'measured')
        self.assertTrue(call_args.kwargs.get('force_refresh'))

    def test_non_numeric_id_not_matched(self):
        """非数字 id 在路由层就被拒（URL 用 <int:>，不匹配 → 404，不到视图）。"""
        self.client.force_login(self.staff)
        # 直接拼 URL（reverse 不接受非数字），验证路由不匹配
        resp = self.client.post(
            '/admin/phm_site/alert/api/diagnose_one/abc/',
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 404)

    def test_zero_id_returns_400(self):
        self.client.force_login(self.staff)
        resp = self.client.post(self._url(0), content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_alert_not_found_returns_404(self):
        self.client.force_login(self.staff)
        c = _make_mock_container(alert_by_id=None)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(self._url(99), content_type='application/json')
        self.assertEqual(resp.status_code, 404)

    def test_diagnosis_service_missing_returns_503(self):
        """container 没有 diagnosis 属性 → 503。"""
        self.client.force_login(self.staff)
        alert = {'id': 5, 'channel': 'C-1', 'alert_type': 'measured',
                 'created_at': 1700000000}
        c = _make_mock_container(alert_by_id=alert)
        # 不设置 c.diagnosis（Mock 默认会自动生成，用 del 模拟缺失）
        del c.diagnosis
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(self._url(5), content_type='application/json')
        self.assertEqual(resp.status_code, 503)


class AlertDiagnoseStatusApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_alert_diagnose_status')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_returns_progress(self):
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn('progress', body)
        self.assertIn('running', body['progress'])


class AlertExportApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_alert_export')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_csv_export(self):
        self.client.force_login(self.staff)
        rows = [{
            'id': 1, 'channel': 'C-1', 'created_at': 1700000000,
            'score': 0.8, 'raw_snapshot': [1, 2, 3], 'ingested_at': 1700000005,
        }]
        c = _make_mock_container(rows=rows)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?format=csv')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp['Content-Type'])
        body = b''.join(resp.streaming_content).decode('utf-8')
        self.assertIn('channel', body)
        self.assertIn('C-1', body)

    def test_json_export(self):
        self.client.force_login(self.staff)
        rows = [{
            'id': 1, 'channel': 'C-1', 'created_at': 1700000000,
            'score': 0.8, 'raw_snapshot': [1], 'ingested_at': 1700000005,
        }]
        c = _make_mock_container(rows=rows)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?format=json')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('application/json', resp['Content-Type'])
        body = json.loads(resp.content)
        self.assertEqual(body['count'], 1)


class AlertCreateApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_alert_create')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_staff_gets_403(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url, {'channel': 'C-1', 'score': 0.9},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 403)

    def test_superuser_success(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1', 'score': 0.9, 'message': 'x'},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['id'], 42)

    def test_missing_channel_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'score': 0.9},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)

    def test_invalid_score_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1', 'score': 'not-a-number'},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)
