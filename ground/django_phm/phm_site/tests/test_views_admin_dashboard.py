"""Page 2: Dashboard tests.

Coverage:
  (a) Anonymous access returns 302 redirect to login.
  (b) staff/superuser access returns 200.
  (c) Renders a placeholder page when the Container is not ready
      (_state.html, no 500 error).
  (d) Three-category classification logic (`_classify_verdict`).
  (e) Time-window boundary calculation (`_window_bounds`: bucket count
      and span for each of today / 7d / 30d).
  (f) Bucket assignment and aggregation (`_collect_dashboard_metrics`:
      counts + bucket distribution + out-of-window discard).

Tests use Django TestCase plus force_login. The Container is mocked so
there is no dependency on a real PHM backend.
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
# Pure-function tests (no Django DB / Container dependency)
# ════════════════════════════════════════════════════════════════════════════

class ClassifyVerdictTest(TestCase):
    """Tests for the three-category classification logic in `_classify_verdict`."""

    def test_human_takes_priority(self):
        """Human annotation takes priority over the LLM verdict (consistent with ``AlertRecord.final_status``)."""
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
        """The first entry in ``VERDICT_CHOICES`` is ``''`` (unannotated), which is synonymous with ``None``."""
        self.assertEqual(_classify_verdict('', 'real'), 'llm')


class HealthTierTest(TestCase):
    """Tests for the health-tier bucketing logic in `_health_tier`."""

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
        """A score of exactly 0.80 maps to *normal*, and 0.50 maps to *warning* (both use >= threshold)."""
        self.assertEqual(_health_tier(0.80)[0], 'normal')
        self.assertEqual(_health_tier(0.50)[0], 'warning')
        self.assertEqual(_health_tier(0.00)[0], 'danger')


class WindowBoundsTest(TestCase):
    """Tests for the time-window boundary calculation in `_window_bounds`."""

    def setUp(self):
        # Fixed now: 2026-07-21 14:30:00 local time
        self.now_dt = dt.datetime(2026, 7, 21, 14, 30, 0)
        self.now = self.now_dt.timestamp()

    def test_today_returns_24_hour_buckets(self):
        start, end, kind, count = _window_bounds('today', self.now)
        self.assertEqual(kind, 'hour')
        self.assertEqual(count, 24)
        # start is today at 00:00:00
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
        # 7d start is 6 days before 7-21 = 7-15
        self.assertEqual((start_dt.year, start_dt.month, start_dt.day),
                         (2026, 7, 15))

    def test_30d_returns_30_day_buckets(self):
        start, end, kind, count = _window_bounds('30d', self.now)
        self.assertEqual(kind, 'day')
        self.assertEqual(count, 30)
        start_dt = dt.datetime.fromtimestamp(start)
        # 30d start is 29 days before 7-21 = 6-22
        self.assertEqual((start_dt.year, start_dt.month, start_dt.day),
                         (2026, 6, 22))

    def test_unknown_window_falls_back_to_today(self):
        """An unknown window value falls back to 'today' semantics instead of raising."""
        start, _end, kind, count = _window_bounds('invalid', self.now)
        self.assertEqual(kind, 'hour')
        self.assertEqual(count, 24)

    def test_end_is_now(self):
        """The end timestamp should equal the passed-in *now* value."""
        for w in ('today', '7d', '30d'):
            _start, end, _kind, _count = _window_bounds(w, self.now)
            self.assertEqual(end, self.now)


class BucketIndexTest(TestCase):
    """Tests for the bucket-position calculation in `_bucket_index`."""

    def test_hour_bucket_within_same_day(self):
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        # 14:30 should land in bucket 14
        ts_1430 = dt.datetime(2026, 7, 21, 14, 30).timestamp()
        self.assertEqual(_bucket_index(ts_1430, today, 'hour'), 14)

    def test_day_bucket(self):
        start = dt.datetime(2026, 6, 22, 0, 0, 0).timestamp()  # 30d start
        # 7-21 should be bucket 29 (6-22 is bucket 0)
        ts_0721 = dt.datetime(2026, 7, 21, 12, 0).timestamp()
        self.assertEqual(_bucket_index(ts_0721, start, 'day'), 29)

    def test_before_start_returns_negative(self):
        """A timestamp before the window start returns a negative index, which the caller discards via range check."""
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        yesterday = dt.datetime(2026, 7, 20, 12, 0).timestamp()
        self.assertLess(_bucket_index(yesterday, today, 'hour'), 0)


class FormatBucketLabelTest(TestCase):
    """Tests for the short-axis label in `_format_bucket_label` (avoids truncation on the x-axis)."""

    def test_hour_label_is_just_hour_number(self):
        """Hour buckets display only the plain hour number ('14' rather than '14:00') to keep 24 buckets legible."""
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        self.assertEqual(_format_bucket_label(14, 'hour', today), '14')
        self.assertEqual(_format_bucket_label(0, 'hour', today), '0')
        self.assertEqual(_format_bucket_label(23, 'hour', today), '23')

    def test_day_label_is_just_day_number(self):
        """Day buckets display only the plain day number ('21' rather than '07-21')."""
        start = dt.datetime(2026, 6, 22, 0, 0, 0).timestamp()
        self.assertEqual(_format_bucket_label(0, 'day', start), '22')   # 6-22
        self.assertEqual(_format_bucket_label(29, 'day', start), '21')  # 7-21

    def test_hour_label_short_enough_no_truncation(self):
        """Short labels are at most 2 characters long so that even 24 buckets display without truncation ('...')."""
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        for i in range(24):
            label = _format_bucket_label(i, 'hour', today)
            self.assertLessEqual(len(label), 2)


class FormatBucketTitleTest(TestCase):
    """Tests for the hover tooltip title in `_format_bucket_title` (full date/time)."""

    def test_hour_title_full_datetime(self):
        today = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        title = _format_bucket_title(14, 'hour', today)
        self.assertEqual(title, '2026-07-21 14:00')

    def test_day_title_full_date(self):
        start = dt.datetime(2026, 6, 22, 0, 0, 0).timestamp()
        title = _format_bucket_title(0, 'day', start)
        self.assertEqual(title, '2026-06-22')


class CollectDashboardMetricsTest(TestCase):
    """Tests for the aggregation logic in `_collect_dashboard_metrics` (using duck-typed alert objects)."""

    def _make_alert(self, ts, human=None, llm=None):
        """Construct a minimal alert object (only needs ``created_at``, ``human_verdict``, ``llm_verdict``)."""
        m = mock.Mock()
        m.created_at = ts
        m.human_verdict = human
        m.llm_verdict = llm
        return m

    def setUp(self):
        # Fixed now: 2026-07-21 14:30
        self.now = dt.datetime(2026, 7, 21, 14, 30).timestamp()

    def test_empty_alerts_returns_zero_counts_and_buckets(self):
        m = _collect_dashboard_metrics('today', [], )
        # `_collect_dashboard_metrics` internally calls the real ``now``, but we do
        # not pass a ``now`` parameter here; with the 'today' window and empty alerts,
        # all three counts should be zero and the bucket list should have 24 entries.
        self.assertEqual(m['counts'], {'human': 0, 'llm': 0, 'undiagnosed': 0, 'total': 0})
        self.assertEqual(len(m['buckets']), 24)
        self.assertTrue(all(b['count'] == 0 for b in m['buckets']))

    def test_classification_three_buckets(self):
        """Each of the three diagnostic statuses is counted into its own bucket."""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        # 3 alerts: human / llm / undiagnosed
        alerts = [
            self._make_alert(today_start + 3600 * 10, human='real'),
            self._make_alert(today_start + 3600 * 11, llm='false_alarm'),
            self._make_alert(today_start + 3600 * 12, human=None, llm=None),
        ]
        # Directly patching the ``now`` parameter of `_window_bounds` would
        # require refactoring `_collect_dashboard_metrics` to accept ``now``.
        # Since the current signature does not accept ``now``, `_window_bounds`
        # falls back to ``_time.time()``.  To keep the pure function testable,
        # we monkeypatch ``_time.time`` instead.
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        self.assertEqual(m['counts']['human'], 1)
        self.assertEqual(m['counts']['llm'], 1)
        self.assertEqual(m['counts']['undiagnosed'], 1)
        self.assertEqual(m['counts']['total'], 3)

    def test_breakdown_real_false_uncertain(self):
        """Breakdown splits each category's verdict into real / false_alarm / uncertain."""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            # human category: one of each verdict
            self._make_alert(today_start + 3600 * 1, human='real'),
            self._make_alert(today_start + 3600 * 2, human='false_alarm'),
            self._make_alert(today_start + 3600 * 3, human='uncertain'),
            # llm category: two verdicts
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
        """When ``human_verdict`` is non-empty, the classification and breakdown both go to the human category."""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            # Both human and llm verdict present -> classified as human, verdict taken from human
            self._make_alert(today_start + 3600 * 1, human='real', llm='false_alarm'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        self.assertEqual(m['counts']['human'], 1)
        self.assertEqual(m['counts']['llm'], 0)
        # breakdown: human.real += 1, llm completely untouched
        self.assertEqual(m['breakdown']['human']['real'], 1)
        self.assertEqual(m['breakdown']['llm']['false_alarm'], 0)

    def test_breakdown_invalid_verdict_ignored(self):
        """A verdict value outside real/false_alarm/uncertain is excluded from the breakdown
        (but still counted normally)."""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            self._make_alert(today_start + 3600 * 1, human='garbage'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        # counts.human = 1 (has human_verdict)
        self.assertEqual(m['counts']['human'], 1)
        # but breakdown.human has all three fields at 0 ('garbage' is unrecognized)
        self.assertEqual(sum(m['breakdown']['human'].values()), 0)

    def test_buckets_have_title_field(self):
        """Every bucket should have a ``title`` field (hover tooltip with the full time)."""
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', [])
        for b in m['buckets']:
            self.assertIn('title', b)
            self.assertIn('label', b)
            self.assertIn('count', b)
        # Bucket 0's title should start with today's date
        self.assertTrue(m['buckets'][0]['title'].startswith('2026-07-21'))

    def test_buckets_have_parts_matrix(self):
        """Each bucket's ``parts`` contains the three dimensions human/llm/undiagnosed (source x verdict 2x3 + undiagnosed count)."""
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', [])
        for b in m['buckets']:
            self.assertIn('parts', b)
            parts = b['parts']
            self.assertIn('human', parts)
            self.assertIn('llm', parts)
            self.assertIn('undiagnosed', parts)
            # human/llm each contain 3 verdict keys
            for src in ('human', 'llm'):
                self.assertEqual(set(parts[src].keys()),
                                 {'real', 'false_alarm', 'uncertain'})
            # undiagnosed is an int (no verdict breakdown)
            self.assertIsInstance(parts['undiagnosed'], int)

    def test_bucket_parts_classification_correct(self):
        """Alerts are distributed into the correct parts segments of the right bucket."""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        alerts = [
            # 10:00: one human real + one llm false_alarm + one undiagnosed
            self._make_alert(today_start + 3600 * 10, human='real'),
            self._make_alert(today_start + 3600 * 10, llm='false_alarm'),
            self._make_alert(today_start + 3600 * 10),
            # 11:00: one human uncertain
            self._make_alert(today_start + 3600 * 11, human='uncertain'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        # 10-o'clock bucket: 3 alerts total
        b10 = m['buckets'][10]
        self.assertEqual(b10['count'], 3)
        self.assertEqual(b10['parts']['human']['real'], 1)
        self.assertEqual(b10['parts']['llm']['false_alarm'], 1)
        self.assertEqual(b10['parts']['undiagnosed'], 1)
        # 11-o'clock bucket: 1 human uncertain
        b11 = m['buckets'][11]
        self.assertEqual(b11['count'], 1)
        self.assertEqual(b11['parts']['human']['uncertain'], 1)

    def test_bucket_parts_sum_equals_count(self):
        """The sum of each bucket's parts segments equals its count (data consistency)."""
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
        """Alerts fall into the correct bucket based on their timestamp."""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        # 10:00 one alert, 11:30 two alerts
        alerts = [
            self._make_alert(today_start + 3600 * 10),
            self._make_alert(today_start + 3600 * 11 + 1800),
            self._make_alert(today_start + 3600 * 11 + 1800),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        self.assertEqual(m['buckets'][10]['count'], 1)
        self.assertEqual(m['buckets'][11]['count'], 2)
        # All other buckets should be zero
        zero_others = sum(1 for i, b in enumerate(m['buckets'])
                          if i not in (10, 11) and b['count'] == 0)
        self.assertEqual(zero_others, 22)

    def test_out_of_window_dropped(self):
        """Alerts earlier than the window start are discarded and not counted in any bucket or total."""
        today_start = dt.datetime(2026, 7, 21, 0, 0, 0).timestamp()
        yesterday = today_start - 3600 * 12  # yesterday noon
        in_window = today_start + 3600 * 5    # today 5:00
        alerts = [
            self._make_alert(yesterday, human='real'),
            self._make_alert(in_window, human='real'),
        ]
        with mock.patch('phm_site.views_admin._time.time', return_value=self.now):
            m = _collect_dashboard_metrics('today', alerts)
        # The alert from yesterday was discarded
        self.assertEqual(m['counts']['total'], 1)
        self.assertEqual(m['counts']['human'], 1)

    def test_30d_window_uses_day_buckets(self):
        """The 30d window should produce 30 day-based buckets."""
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
# View-layer tests (mocked Container)
# ════════════════════════════════════════════════════════════════════════════

class DashboardViewAccessTest(TestCase):
    """Tests for page access permissions and rendering."""

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
        """When the Container is not ready, the view renders the ``_state.html`` placeholder page instead of raising a 500 error."""
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
    """Tests for the window GET parameter and banner/card rendering."""

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
        """Simulate the return value of ``services_bridge.get_link_status``."""
        return {'status': status, 'rtt_ms': rtt_ms, 'last_success_ts': _time.time()}

    def test_default_window_is_today(self):
        """Without a ``window`` parameter, the view defaults to 'today'."""
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as sb, \
             mock.patch('phm_site.views_admin.AlertRecord') as ar_qs:
            sb.get_state.return_value = 'ready'
            sb.get_container.return_value = self._mock_container()
            ar_qs.objects.filter.return_value.only.return_value = []
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # today tab should be marked active
        self.assertContains(resp, '今天</span>')  # active span closing tag

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
        """An invalid ``window`` value does not raise; it falls back to 'today'."""
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
        """Different health scores should render the corresponding banner tier class."""
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
        """Key text for all three stat cards should be rendered."""
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
    """When ``system_health()`` raises, the view degrades to 1.0 instead of returning 500."""

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
        # Degrades to 1.0 (normal tier)
        self.assertContains(resp, 'phm-dash-banner-normal')


class DashboardViewAlertsErrorTest(TestCase):
    """When the ``AlertRecord`` query raises, the view degrades to empty metrics instead of returning 500."""

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
            # The query chain raises an error
            ar_qs.objects.filter.side_effect = RuntimeError("db locked")
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # Falls through to the empty-metrics branch: shows "no alerts in this time window"
        self.assertContains(resp, '当前时间窗内暂无告警记录')


class DashboardViewLinkStatusTest(TestCase):
    """Tests for the link latency and status badge on the right side of the banner (replaces the old "anomaly threshold")."""

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
        """The banner renders the link latency text (no longer the "anomaly threshold")."""
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
        # No longer renders "anomaly threshold"
        self.assertNotContains(resp, '异常阈值')

    def test_link_status_badge_rendered(self):
        """The link-status badge ``phm-dash-link-<status>`` is rendered correctly."""
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
        """When ``rtt_ms`` is ``None``, the template renders an em-dash placeholder."""
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
        # Em-dash placeholder (template uses the -- entity)
        self.assertContains(resp, '—')

    def test_get_link_status_failure_degrades_to_unknown(self):
        """When ``services_bridge.get_link_status`` raises, the view degrades to ``latency=None`` and ``status=unknown``."""
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
    """Tests for the breakdown rendering on the three stat cards (real / false_alarm / uncertain)."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_three_stat_cards_with_breakdown(self):
        """All three cards should render the real / false_alarm / uncertain breakdown rows (the first two cards have breakdowns; the third does not)."""
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
        # Human card breakdown
        self.assertContains(resp, '实警')
        self.assertContains(resp, '虚警')
        self.assertContains(resp, '待定')


class DashboardViewAutoRefreshTest(TestCase):
    """Tests for the auto-refresh page reload feature (``?auto=1``)."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_dashboard')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def _setup_mocks(self):
        """Returns a ``(services_bridge mock, AlertRecord mock)`` context-manager tuple for setup."""
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
        """Without the ``?auto`` parameter, auto-refresh is checked by default (per requirements feedback: dashboard should auto-refresh by default)."""
        self.client.force_login(self.staff)
        self._setup_mocks()
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode('utf-8')
        # checkbox should be checked
        toggle_part = body.split('id="phm-auto-refresh-toggle"')[1].split('>')[0]
        self.assertIn('checked', toggle_part)

    def test_auto_param_zero_disables_refresh(self):
        """``?auto=0`` explicitly disables auto-refresh."""
        self.client.force_login(self.staff)
        self._setup_mocks()
        resp = self.client.get(self.url, {'auto': '0'})
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode('utf-8')
        toggle_part = body.split('id="phm-auto-refresh-toggle"')[1].split('>')[0]
        self.assertNotIn('checked', toggle_part)

    def test_auto_param_enables_refresh(self):
        """When ``?auto=1``, the checkbox is checked and the countdown script is rendered."""
        self.client.force_login(self.staff)
        self._setup_mocks()
        resp = self.client.get(self.url, {'auto': '1'})
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode('utf-8')
        # checkbox checked
        toggle_part = body.split('id="phm-auto-refresh-toggle"')[1].split('>')[0]
        self.assertIn('checked', toggle_part)
        # Renders the refresh interval (15s)
        self.assertIn('15', body)

    def test_refresh_seconds_rendered_in_template(self):
        """The template renders ``_DASHBOARD_REFRESH_SECONDS``."""
        self.client.force_login(self.staff)
        self._setup_mocks()
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # The auto-refresh toggle label "auto-refresh every 15s"
        self.assertContains(resp, '每 15s 自动刷新')
