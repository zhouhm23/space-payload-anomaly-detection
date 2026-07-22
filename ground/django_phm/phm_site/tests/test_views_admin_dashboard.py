"""第 2 页：仪表盘测试。

覆盖：
  (a) 匿名访问 302 跳登录
  (b) staff/superuser 访问 200
  (c) Container 未就绪时渲染占位页（_state.html，不 500）
  (d) 三分类逻辑（_classify_verdict）
  (e) 时间窗边界（_window_bounds：today/7d/30d 各自的桶数与跨度）
  (f) 桶分配与聚合（_collect_dashboard_metrics：计数 + 桶分布 + 越界丢弃）

测试用 Django TestCase + force_login。Container 走 mock（不依赖真实 PHM）。
"""
from __future__ import annotations

import datetime as dt
import time as _time
from unittest import mock

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm_site.views_admin import (
    _bucket_index,
    _classify_verdict,
    _collect_dashboard_metrics,
    _format_bucket_label,
    _format_bucket_title,
    _health_tier,
    _window_bounds,
)


# ════════════════════════════════════════════════════════════════════════════
# 纯函数测试（不依赖 Django DB / Container）
# ════════════════════════════════════════════════════════════════════════════

class ClassifyVerdictTest(TestCase):
    """_classify_verdict 三分类逻辑。"""

    def test_human_takes_priority(self):
        """人工标注优先于 LLM（与 AlertRecord.final_status 一致）。"""
        self.assertEqual(_classify_verdict('real', 'false_alarm'), 'human')
        self.assertEqual(_classify_verdict('uncertain', 'real'), 'human')

    def test_llm_when_no_human(self):
        self.assertEqual(_classify_verdict('', 'real'), 'llm')
        self.assertEqual(_classify_verdict(None, 'false_alarm'), 'llm')

    def test_undiagnosed_when_both_empty(self):
        self.assertEqual(_classify_verdict('', ''), 'undiagnosed')
        self.assertEqual(_classify_verdict(None, None), 'undiagnosed')
        self.assertEqual(_classify_verdict('', None), 'undiagnosed')

    def test_empty_string_equals_none(self):
        """VERDICT_CHOICES 第一项是 ''（未标注），与 None 同义。"""
        self.assertEqual(_classify_verdict('', 'real'), 'llm')


class HealthTierTest(TestCase):
    """_health_tier 分档逻辑。"""

    def test_normal_above_80(self):
        tier, text = _health_tier(0.80)
        self.assertEqual(tier, 'normal')
        tier, text = _health_tier(1.0)
        self.assertEqual(tier, 'normal')
        tier, text = _health_tier(0.95)
        self.assertEqual(tier, 'normal')

    def test_warning_between_50_and_80(self):
        tier, _ = _health_tier(0.50)
        self.assertEqual(tier, 'warning')
        tier, _ = _health_tier(0.79)
        self.assertEqual(tier, 'warning')

    def test_danger_below_50(self):
        tier, _ = _health_tier(0.00)
        self.assertEqual(tier, 'danger')
        tier, _ = _health_tier(0.49)
        self.assertEqual(tier, 'danger')

    def test_boundary_values(self):
        """0.80 归 normal，0.50 归 warning（>= 阈值）。"""
        self.assertEqual(_health_tier(0.80)[0], 'normal')
        self.assertEqual(_health_tier(0.50)[0], 'warning')
        self.assertEqual(_health_tier(0.00)[0], 'danger')


class WindowBoundsTest(TestCase):
    """_window_bounds 时间窗计算。"""

    def setUp(self):
        # 固定 now：2026-07-21 14:30:00 本地时间
        self.now_dt = dt.datetime(2026, 7, 21, 14, 30, 0)
        self.now = self.now_dt.timestamp()

    def test_today_returns_24_hour_buckets(self):
        start, end, kind, count = _window_bounds('today', self.now)
        self.assertEqual(kind, 'hour')
        self.assertEqual(count, 24)
        # start 是今日 00:00:00
        start_dt = dt.datetime.fromtimestamp(start)
        self.assertEqual(start_dt.hour, 0)
        self.assertEqual(start_dt.minute, 0)
        self.assertEqual((start_dt.year, start_dt.month, start_dt.day),
                         (2026, 7, 21))

    def test_7d_returns_7_day_buckets(self):
        start, end, kind, count = _window_bounds('7d', self.now)
        self.assertEqual(kind, 'day')
        self.assertEqual(count, 7)
        start_dt = dt.datetime.fromtimestamp(start)
        # 7d 起点是 7-21 往前推 6 天 = 7-15
        self.assertEqual((start_dt.year, start_dt.month, start_dt.day),
                         (2026, 7, 15))

    def test_30d_returns_30_day_buckets(self):
        start, end, kind, count = _window_bounds('30d', self.now)
        self.assertEqual(kind, 'day')
        self.assertEqual(count, 30)
        start_dt = dt.datetime.fromtimestamp(start)
        # 30d 起点是 7-21 往前推 29 天 = 6-22
        self.assertEqual((start_dt.year, start_dt.month, start_dt.day),
                         (2026, 6, 22))

    def test_unknown_window_falls_back_to_today(self):
        """未知 window 值兜底为 today 语义（不抛错）。"""
        start, _end, kind, count = _window_bounds('invalid', self.now)
        self.assertEqual(kind, 'hour')
        self.assertEqual(count, 24)

    def test_end_is_now(self):
        """end 时间戳应等于传入的 now。"""
        for w in ('today', '7d', '30d'):
            _start, end, _kind, _count = _window_bounds(w, self.now)
            self.assertEqual(end, self.now)


class BucketIndexTest(TestCase):
    """_bucket_index 桶位置计算。"""

    def test_hour_bucket_within_same_day(self):
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        # 14:30 应落入第 14 桶
        ts_1430 = dt.datetime(2026, 7, 21, 14, 30).timestamp()
        self.assertEqual(_bucket_index(ts_1430, today, 'hour'), 14)

    def test_day_bucket(self):
        start = dt.datetime(2026, 6, 22, 0, 0, 0).timestamp()  # 30d 起点
        # 7-21 应是第 29 桶（6-22 是第 0 桶）
        ts_0721 = dt.datetime(2026, 7, 21, 12, 0).timestamp()
        self.assertEqual(_bucket_index(ts_0721, start, 'day'), 29)

    def test_before_start_returns_negative(self):
        """越界（早于 start）返回负数——上层用范围检查丢弃。"""
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        yesterday = dt.datetime(2026, 7, 20, 12, 0).timestamp()
        self.assertLess(_bucket_index(yesterday, today, 'hour'), 0)


class FormatBucketLabelTest(TestCase):
    """_format_bucket_label 短标签（x 轴刻度用，避免截断）。"""

    def test_hour_label_is_just_hour_number(self):
        """小时桶只显示纯小时数字（'14' 而非 '14:00'），24 桶不挤。"""
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        self.assertEqual(_format_bucket_label(14, 'hour', today), '14')
        self.assertEqual(_format_bucket_label(0, 'hour', today), '0')
        self.assertEqual(_format_bucket_label(23, 'hour', today), '23')

    def test_day_label_is_just_day_number(self):
        """天桶只显示纯日数字（'21' 而非 '07-21'）。"""
        start = dt.datetime(2026, 6, 22, 0, 0, 0).timestamp()
        self.assertEqual(_format_bucket_label(0, 'day', start), '22')   # 6-22
        self.assertEqual(_format_bucket_label(29, 'day', start), '21')  # 7-21

    def test_hour_label_short_enough_no_truncation(self):
        """短标签长度 ≤ 2 字符，24 桶也能完整显示不出现 '...'。"""
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        for i in range(24):
            label = _format_bucket_label(i, 'hour', today)
            self.assertLessEqual(len(label), 2)


class FormatBucketTitleTest(TestCase):
    """_format_bucket_title 悬停 title（完整时间）。"""

    def test_hour_title_full_datetime(self):
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        title = _format_bucket_title(14, 'hour', today)
        self.assertEqual(title, '2026-07-21 14:00')

    def test_day_title_full_date(self):
        start = dt.datetime(2026, 6, 22, 0, 0, 0).timestamp()
        title = _format_bucket_title(0, 'day', start)
        self.assertEqual(title, '2026-06-22')


class CollectDashboardMetricsTest(TestCase):
    """_collect_dashboard_metrics 聚合（用鸭子类型告警）。"""

    def _make_alert(self, ts, human=None, llm=None):
        """构造一个最小告警对象（只需 created_at/human_verdict/llm_verdict）。"""
        m = mock.Mock()
        m.created_at = ts
        m.human_verdict = human
        m.llm_verdict = llm
        return m

    def setUp(self):
        # 固定 now：2026-07-21 14:30
        self.now = dt.datetime(2026, 7, 21, 14, 30).timestamp()

    def test_empty_alerts_returns_zero_counts_and_buckets(self):
        m = _collect_dashboard_metrics('today', [], )
        # _collect_dashboard_metrics 内部用真实 now，但我们这里不传 now 参数；
        # 用 today 窗口，告警空时三计数全 0，桶长 24。
        self.assertEqual(m['counts'], {'human': 0, 'llm': 0, 'undiagnosed': 0, 'total': 0})
        self.assertEqual(len(m['buckets']), 24)
        self.assertTrue(all(b['count'] == 0 for b in m['buckets']))

    def test_classification_three_buckets(self):
        """三种诊断状态分别计数到不同桶。"""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        # 3 条告警：human / llm / undiagnosed
        alerts = [
            self._make_alert(today_start + 3600 * 10, human='real'),
            self._make_alert(today_start + 3600 * 11, llm='false_alarm'),
            self._make_alert(today_start + 3600 * 12, human=None, llm=None),
        ]
        # 直接 patch _window_bounds 的 now 参数：需要重构 _collect_dashboard_metrics
        # 接受 now 参数才行。但当前签名不接 now，所以走 _window_bounds 默认用
        # _time.time()。为保持纯函数可测，这里改用 monkeypatch。
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        self.assertEqual(m['counts']['human'], 1)
        self.assertEqual(m['counts']['llm'], 1)
        self.assertEqual(m['counts']['undiagnosed'], 1)
        self.assertEqual(m['counts']['total'], 3)

    def test_breakdown_real_false_uncertain(self):
        """breakdown 把每类的 verdict 细分成 real/false_alarm/uncertain。"""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            # human 类 3 种 verdict 各 1 条
            self._make_alert(today_start + 3600 * 1, human='real'),
            self._make_alert(today_start + 3600 * 2, human='false_alarm'),
            self._make_alert(today_start + 3600 * 3, human='uncertain'),
            # llm 类 2 种 verdict
            self._make_alert(today_start + 3600 * 4, llm='real'),
            self._make_alert(today_start + 3600 * 5, llm='false_alarm'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        self.assertEqual(m['breakdown']['human'],
                         {'real': 1, 'false_alarm': 1, 'uncertain': 1})
        self.assertEqual(m['breakdown']['llm']['real'], 1)
        self.assertEqual(m['breakdown']['llm']['false_alarm'], 1)
        self.assertEqual(m['breakdown']['llm']['uncertain'], 0)

    def test_breakdown_human_priority_over_llm(self):
        """当 human_verdict 非空时，分类走 human，breakdown 也归 human。"""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            # 同时有 human 和 llm verdict → 归 human，verdict 取 human 的
            self._make_alert(today_start + 3600 * 1, human='real', llm='false_alarm'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        self.assertEqual(m['counts']['human'], 1)
        self.assertEqual(m['counts']['llm'], 0)
        # breakdown：human.real += 1，llm 完全不动
        self.assertEqual(m['breakdown']['human']['real'], 1)
        self.assertEqual(m['breakdown']['llm']['false_alarm'], 0)

    def test_breakdown_invalid_verdict_ignored(self):
        """verdict 值不在 real/false_alarm/uncertain 内时不计入 breakdown
        （但 counts 照常计）。"""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            self._make_alert(today_start + 3600 * 1, human='garbage'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        # counts.human = 1（有 human_verdict）
        self.assertEqual(m['counts']['human'], 1)
        # 但 breakdown.human 三个字段全 0（'garbage' 不认）
        self.assertEqual(sum(m['breakdown']['human'].values()), 0)

    def test_buckets_have_title_field(self):
        """每个桶都应有 title 字段（悬停完整时间）。"""
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', [])
        for b in m['buckets']:
            self.assertIn('title', b)
            self.assertIn('label', b)
            self.assertIn('count', b)
        # 第 0 桶 title 应是今天的日期开头
        self.assertTrue(m['buckets'][0]['title'].startswith('2026-07-21'))

    def test_buckets_have_parts_matrix(self):
        """每桶 parts 含 human/llm/undiagnosed 三维（来源×verdict 2×3 + undiagnosed 计数）。"""
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', [])
        for b in m['buckets']:
            self.assertIn('parts', b)
            parts = b['parts']
            self.assertIn('human', parts)
            self.assertIn('llm', parts)
            self.assertIn('undiagnosed', parts)
            # human/llm 各含 3 个 verdict key
            for src in ('human', 'llm'):
                self.assertEqual(set(parts[src].keys()),
                                 {'real', 'false_alarm', 'uncertain'})
            # undiagnosed 是 int（无 verdict 细分）
            self.assertIsInstance(parts['undiagnosed'], int)

    def test_bucket_parts_classification_correct(self):
        """告警分到正确桶的 correct parts 段。"""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            # 10:00：human real 一条 + llm false 一条 + 未诊断一条
            self._make_alert(today_start + 3600 * 10, human='real'),
            self._make_alert(today_start + 3600 * 10, llm='false_alarm'),
            self._make_alert(today_start + 3600 * 10),
            # 11:00：human uncertain 一条
            self._make_alert(today_start + 3600 * 11, human='uncertain'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        # 10 点桶：3 条总数
        b10 = m['buckets'][10]
        self.assertEqual(b10['count'], 3)
        self.assertEqual(b10['parts']['human']['real'], 1)
        self.assertEqual(b10['parts']['llm']['false_alarm'], 1)
        self.assertEqual(b10['parts']['undiagnosed'], 1)
        # 11 点桶：1 条 human uncertain
        b11 = m['buckets'][11]
        self.assertEqual(b11['count'], 1)
        self.assertEqual(b11['parts']['human']['uncertain'], 1)

    def test_bucket_parts_sum_equals_count(self):
        """每桶 parts 各段加起来 = count（数据一致性）。"""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            self._make_alert(today_start + 3600 * 5, human='real'),
            self._make_alert(today_start + 3600 * 5, llm='false_alarm'),
            self._make_alert(today_start + 3600 * 5, llm='uncertain'),
            self._make_alert(today_start + 3600 * 5),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        b5 = m['buckets'][5]
        parts_sum = sum(b5['parts']['human'].values()) + \
                    sum(b5['parts']['llm'].values()) + \
                    b5['parts']['undiagnosed']
        self.assertEqual(parts_sum, b5['count'])

    def test_bucket_distribution(self):
        """告警按时间正确落到对应桶。"""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        # 10:00 一条，11:30 两条
        alerts = [
            self._make_alert(today_start + 3600 * 10),
            self._make_alert(today_start + 3600 * 11 + 1800),
            self._make_alert(today_start + 3600 * 11 + 1800),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        self.assertEqual(m['buckets'][10]['count'], 1)
        self.assertEqual(m['buckets'][11]['count'], 2)
        # 其他桶全 0
        zero_others = sum(1 for i, b in enumerate(m['buckets'])
                          if i not in (10, 11) and b['count'] == 0)
        self.assertEqual(zero_others, 22)

    def test_out_of_window_dropped(self):
        """早于 start_ts 的告警被丢弃，不计入桶也不计入计数。"""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        yesterday = today_start - 3600 * 12  # 昨天中午
        in_window = today_start + 3600 * 5    # 今天 5:00
        alerts = [
            self._make_alert(yesterday, human='real'),
            self._make_alert(in_window, human='real'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        # 昨天那条被丢弃
        self.assertEqual(m['counts']['total'], 1)
        self.assertEqual(m['counts']['human'], 1)

    def test_30d_window_uses_day_buckets(self):
        """30d 窗口应得到 30 个按天的桶。"""
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('30d', [])
        self.assertEqual(m['bucket_kind'], 'day')
        self.assertEqual(len(m['buckets']), 30)

    def test_7d_window_uses_day_buckets(self):
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('7d', [])
        self.assertEqual(m['bucket_kind'], 'day')
        self.assertEqual(len(m['buckets']), 7)


# ════════════════════════════════════════════════════════════════════════════
# 视图层测试（mock Container）
# ════════════════════════════════════════════════════════════════════════════

class DashboardViewAccessTest(TestCase):
    """页面访问权限与渲染。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_anonymous_redirects_to_login(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/admin/login/', resp['Location'])

    def test_staff_can_access(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                health=mock.Mock(system_health=mock.Mock(
                    return_value={'system': 0.85, 'channels': {'a': 0.8},
                                  'threshold': 0.3}
                ))
            )
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '仪表盘')

    def test_superuser_can_access(self):
        self.client.force_login(self.superuser)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = mock.Mock(
                health=mock.Mock(system_health=mock.Mock(
                    return_value={'system': 0.85, 'channels': {}, 'threshold': 0.3}
                ))
            )
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_container_not_ready_renders_state_page(self):
        """Container 未就绪时渲染 _state.html 占位页，不 500。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'initializing'
            sb.get_init_error.return_value = None
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '正在初始化')

    def test_container_failed_shows_error(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb:
            sb.get_state.return_value = 'failed'
            sb.get_init_error.return_value = 'torch import failed'
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'torch import failed')


class DashboardViewWindowTest(TestCase):
    """window GET 参数处理与 banner/卡片渲染。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def _mock_container(self, system=0.85):
        container = mock.Mock()
        container.health.system_health.return_value = {
            'system': system, 'channels': {'a': 0.8, 'b': 0.9},
        }
        return container

    def _mock_link_status(self, status='online', rtt_ms=2000.0):
        """模拟 services_bridge.get_link_status 返回值。"""
        return {'status': status, 'rtt_ms': rtt_ms, 'last_success_ts': _time.time()}

    def test_default_window_is_today(self):
        """无 window 参数走默认 today。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container()
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # today tab 应标记 active
        self.assertContains(resp, '今天</span>')  # active span 闭合

    def test_window_7d_param_accepted(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container()
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url, {'window': '7d'})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '最近 7 天</span>')

    def test_window_30d_param_accepted(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container()
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url, {'window': '30d'})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '最近 30 天</span>')

    def test_invalid_window_falls_back_to_today(self):
        """非法 window 值不抛错，兜底为 today。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container()
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url, {'window': 'garbage'})
        self.assertEqual(resp.status_code, 200)
        # today tab active
        self.assertContains(resp, '今天</span>')

    def test_banner_health_tier_rendered(self):
        """不同健康度应渲染对应 banner tier 类名。"""
        self.client.force_login(self.staff)
        cases = [
            (0.95, 'normal'),
            (0.60, 'warning'),
            (0.20, 'danger'),
        ]
        for system_val, tier in cases:
            with mock.patch('phm_site.views_admin.services_bridge') as sb, \
                 mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
                sb.get_state.return_value = 'ready'
                sb.get_container.return_value = self._mock_container(system=system_val)
                ar_qs.objects.filter.return_value.only.return_value = []
                resp = self.client.get(self.url)
            self.assertEqual(resp.status_code, 200)
            self.assertContains(resp, f'phm-dash-banner-{tier}')

    def test_three_stat_cards_rendered(self):
        """三张统计卡片的关键文字都应渲染。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container()
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertContains(resp, '已人工诊断')
        self.assertContains(resp, 'LLM 诊断')
        self.assertContains(resp, '未诊断')


class DashboardViewHealthErrorTest(TestCase):
    """system_health() 抛错时降级为 1.0 不 500。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_health_failure_degrades_gracefully(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            bad_container = mock.Mock()
            bad_container.health.system_health.side_effect = RuntimeError("boom")
            sb.get_container.return_value = bad_container
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # 降级到 1.0（normal 分档）
        self.assertContains(resp, 'phm-dash-banner-normal')


class DashboardViewAlertsErrorTest(TestCase):
    """AlertRecord 查询抛错时降级为空 metrics 不 500。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_alerts_query_failure_degrades_gracefully(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            container = mock.Mock()
            container.health.system_health.return_value = {
                'system': 0.85, 'channels': {},
            }
            sb.get_container.return_value = container
            sb.get_link_status.return_value = {
                'status': 'online', 'rtt_ms': 2000.0, 'last_success_ts': _time.time(),
            }
            # 查询链抛错
            ar_qs.objects.filter.side_effect = RuntimeError("db locked")
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # 走空 metrics 分支：显示"暂无告警记录"
        self.assertContains(resp, '当前时间窗内暂无告警记录')


class DashboardViewLinkStatusTest(TestCase):
    """banner 右侧显示天地延迟 + 链路状态（替代之前的"异常阈值"）。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def _container(self, system=0.85):
        c = mock.Mock()
        c.health.system_health.return_value = {
            'system': system, 'channels': {'a': 0.8, 'b': 0.9},
        }
        return c

    def test_link_latency_displayed_in_banner(self):
        """banner 渲染"天地延迟 Xms"字样（不再是"异常阈值"）。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._container()
            sb.get_link_status.return_value = {
                'status': 'online', 'rtt_ms': 2000.0,
                'last_success_ts': _time.time(),
            }
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '天地延迟')
        self.assertContains(resp, '2000')
        self.assertContains(resp, 'ms')
        # 不再渲染"异常阈值"
        self.assertNotContains(resp, '异常阈值')

    def test_link_status_badge_rendered(self):
        """链路状态徽章 phm-dash-link-<status> 渲染。"""
        self.client.force_login(self.staff)
        for status in ['online', 'degraded', 'offline', 'waiting']:
            with mock.patch('phm_site.views_admin.services_bridge') as sb, \
                 mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
                sb.get_state.return_value = 'ready'
                sb.get_container.return_value = self._container()
                sb.get_link_status.return_value = {
                    'status': status, 'rtt_ms': None,
                    'last_success_ts': _time.time(),
                }
                ar_qs.objects.filter.return_value.only.return_value = []
                resp = self.client.get(self.url)
            self.assertEqual(resp.status_code, 200)
            self.assertContains(resp, f'phm-dash-link-{status}')

    def test_latency_none_renders_dash(self):
        """rtt_ms 为 None 时显示"—"。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._container()
            sb.get_link_status.return_value = {
                'status': 'waiting', 'rtt_ms': None,
                'last_success_ts': _time.time(),
            }
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # "—" 占位（模板用 — 实体）
        self.assertContains(resp, '—')

    def test_get_link_status_failure_degrades_to_unknown(self):
        """services_bridge.get_link_status 抛错时降级 latency=None / status=unknown。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._container()
            sb.get_link_status.side_effect = RuntimeError("bridge boom")
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'phm-dash-link-unknown')


class DashboardViewBreakdownRenderTest(TestCase):
    """三卡片细分（实警/虚警/待定）渲染。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_three_stat_cards_with_breakdown(self):
        """三卡片都应渲染实警/虚警/待定细分行（前两张有细分，第三张无）。"""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            container = mock.Mock()
            container.health.system_health.return_value = {
                'system': 0.85, 'channels': {'a': 0.8},
            }
            sb.get_container.return_value = container
            sb.get_link_status.return_value = {
                'status': 'online', 'rtt_ms': 100.0,
                'last_success_ts': _time.time(),
            }
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # 人工卡片细分
        self.assertContains(resp, '实警')
        self.assertContains(resp, '虚警')
        self.assertContains(resp, '待定')


class DashboardViewAutoRefreshTest(TestCase):
    """自动刷新（?auto=1）页面级 reload。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def _setup_mocks(self):
        """返回 (services_bridge mock, AlertRecord mock) 上下文管理器元组。"""
        sb_patch = mock.patch('phm_site.views_admin.services_bridge')
        ar_patch = mock.patch('phm_site.views_admin.AlertRecord')
        sb = sb_patch.start()
        ar_qs = ar_patch.start()
        self.addCleanup(sb_patch.stop)
        self.addCleanup(ar_patch.stop)
        sb.get_state.return_value = 'ready'
        container = mock.Mock()
        container.health.system_health.return_value = {
            'system': 0.85, 'channels': {'a': 0.8},
        }
        sb.get_container.return_value = container
        sb.get_link_status.return_value = {
            'status': 'online', 'rtt_ms': 100.0,
            'last_success_ts': _time.time(),
        }
        ar_qs.objects.filter.return_value.only.return_value = []
        return sb, ar_qs

    def test_default_on_no_auto_param(self):
        """无 ?auto 参数时默认勾选自动刷新（需求书反馈：dashboard 默认开自动刷新）。"""
        self.client.force_login(self.staff)
        self._setup_mocks()
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode('utf-8')
        # checkbox 应带 checked
        toggle_part = body.split('id="phm-auto-refresh-toggle"')[1].split('>')[0]
        self.assertIn('checked', toggle_part)

    def test_auto_param_zero_disables_refresh(self):
        """?auto=0 显式关闭自动刷新。"""
        self.client.force_login(self.staff)
        self._setup_mocks()
        resp = self.client.get(self.url, {'auto': '0'})
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode('utf-8')
        toggle_part = body.split('id="phm-auto-refresh-toggle"')[1].split('>')[0]
        self.assertNotIn('checked', toggle_part)

    def test_auto_param_enables_refresh(self):
        """?auto=1 时 checkbox 勾选 + 渲染倒计时脚本。"""
        self.client.force_login(self.staff)
        self._setup_mocks()
        resp = self.client.get(self.url, {'auto': '1'})
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode('utf-8')
        # checkbox checked
        toggle_part = body.split('id="phm-auto-refresh-toggle"')[1].split('>')[0]
        self.assertIn('checked', toggle_part)
        # 渲染了刷新间隔（15s）
        self.assertIn('15', body)

    def test_refresh_seconds_rendered_in_template(self):
        """模板里渲染了 _DASHBOARD_REFRESH_SECONDS。"""
        self.client.force_login(self.staff)
        self._setup_mocks()
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # 自动刷新开关的文案 "每 15s 自动刷新"
        self.assertContains(resp, '每 15s 自动刷新')
