"""Page 7: telemetry data management tests.

Coverage:
  (a) Helpers: _parse_tel_limit / _decorate_tel_rows / _list_tel_channels
  (b) telemetry_view access: anonymous / staff / superuser / Container-not-ready
      placeholder / empty-state (no channel) / pagination / time filtering
  (c) AJAX endpoints: create / delete / export / channels — covering
      permissions (403/302) + business logic (success / validation /
      service not ready)
  (d) Decorated rows carry the type_label / origin_label / utc_time used by
      the template and the ECharts data injection (rows_json).
"""
from __future__ import annotations

import json
from unittest import mock

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm_site.views_admin import (
    _decorate_tel_rows,
    _parse_tel_limit,
    _TEL_PAGE_SIZE_OPTIONS,
)


# ════════════════════════════════════════════════════════════════════════════
# Pure-helper tests
# ════════════════════════════════════════════════════════════════════════════

class ParseTelLimitTest(TestCase):

    def test_default(self):
        self.assertEqual(_parse_tel_limit(None), 20)
        self.assertEqual(_parse_tel_limit(''), 20)

    def test_valid_int(self):
        self.assertEqual(_parse_tel_limit('50'), 50)

    def test_clamp_min(self):
        self.assertEqual(_parse_tel_limit('0'), 1)
        self.assertEqual(_parse_tel_limit('-5'), 1)

    def test_clamp_max(self):
        self.assertEqual(_parse_tel_limit('5000'), 1000)

    def test_invalid_falls_back(self):
        self.assertEqual(_parse_tel_limit('abc'), 20)


class DecorateTelRowsTest(TestCase):
    """The decorator merges raw_value / predicted_value and tags manual rows."""

    def _row(self, **over):
        base = {
            'id': 1, 'timestamp': 1700000000.0, 'raw_value': None,
            'anomaly_score': None, 'predicted_value': None,
            'predicted_anomaly_score': None, 'origin_ts': None,
            'ingested_at': 1700000000, 'deleted_at': None, 'origin': 'acq',
            'channel': 'C-1',
        }
        base.update(over)
        return base

    def test_raw_only_labelled_real(self):
        out = _decorate_tel_rows([self._row(raw_value=1.5)])
        self.assertEqual(out[0]['type_label'], '真实')
        self.assertIn('1.500', out[0]['value_display'])
        self.assertEqual(out[0]['origin_label'], '采集')
        self.assertFalse(out[0]['is_manual'])

    def test_predicted_only_labelled_predicted(self):
        out = _decorate_tel_rows([self._row(predicted_value=2.5)])
        self.assertEqual(out[0]['type_label'], '预测')

    def test_both_labelled_joint(self):
        out = _decorate_tel_rows([self._row(raw_value=1.5, predicted_value=2.5)])
        self.assertEqual(out[0]['type_label'], '真实+预测')
        self.assertIn('真实', out[0]['value_display'])
        self.assertIn('预测', out[0]['value_display'])

    def test_manual_origin_marked(self):
        out = _decorate_tel_rows([self._row(raw_value=1.0, origin='manual')])
        self.assertTrue(out[0]['is_manual'])
        self.assertIn('手动', out[0]['origin_label'])

    def test_utc_time_string(self):
        out = _decorate_tel_rows([self._row(timestamp=1700000000.0)])
        self.assertIsInstance(out[0]['utc_time'], str)
        self.assertIn('UTC', out[0]['utc_time'])

    def test_invalid_timestamp_falls_back(self):
        out = _decorate_tel_rows([self._row(timestamp='not-a-number')])
        self.assertEqual(out[0]['utc_time'], '—')


# ════════════════════════════════════════════════════════════════════════════
# Mock-container helpers
# ════════════════════════════════════════════════════════════════════════════

def _make_mock_container(rows=None, count=None):
    """Build a mock Container with sqlite / config wired for telemetry_view."""
    c = mock.Mock()
    c.sqlite.query_tel_page.return_value = rows or []
    c.sqlite.count_tel.return_value = (count if count is not None else len(rows or []))
    c.sqlite.soft_delete_tel.return_value = 1
    c.sqlite.insert_tel_manual.return_value = True
    c.sqlite.iter_tel_rows.return_value = iter(rows or [])
    c.sqlite.enabled = True
    c.sqlite._conn = mock.Mock()
    # _list_tel_channels: no on-disk tables by default.
    c.sqlite._conn.execute.return_value.fetchall.return_value = []
    c.config.load.return_value = {'device_tree': [
        {'type': 'sensor', 'name': '传感器 C-1', 'channelName': 'C-1', 'unit': 'A'},
    ]}
    return c


def _patch_container(c):
    """Patch services_bridge so the three-state machine takes the ready branch."""
    return (
        mock.patch('phm_site.services_bridge.get_container', return_value=c),
        mock.patch('phm_site.services_bridge.get_state', return_value='ready'),
    )


# ════════════════════════════════════════════════════════════════════════════
# telemetry_view access + pagination + filtering
# ════════════════════════════════════════════════════════════════════════════

class TelemetryViewAccessTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_telemetry')
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

    def test_container_not_ready_renders_state_page(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.services_bridge.get_state',
                        return_value='initializing'):
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_empty_state_when_no_channel(self):
        """No channel selected → renders the 'please select a sensor' prompt."""
        self.client.force_login(self.staff)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '请先选择一个传感器')
        # No query should have run.
        c.sqlite.query_tel_page.assert_not_called()

    def test_staff_can_access_with_channel(self):
        self.client.force_login(self.staff)
        c = _make_mock_container(rows=[
            {'id': 1, 'timestamp': 1700000000.0, 'raw_value': 1.0,
             'predicted_value': None, 'origin': 'acq', 'anomaly_score': None,
             'predicted_anomaly_score': None, 'origin_ts': None,
             'ingested_at': 1700000000, 'deleted_at': None, 'channel': 'C-1'},
        ])
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?channel=C-1')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '遥测数据管理')
        self.assertContains(resp, '真实')  # type_label rendered

    def test_channel_forwarded_to_query(self):
        self.client.force_login(self.staff)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            self.client.get(self.url + '?channel=C-1')
        args, kwargs = c.sqlite.query_tel_page.call_args
        self.assertEqual(args[0], 'C-1')

    def test_time_window_forwarded_to_query(self):
        self.client.force_login(self.staff)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            self.client.get(self.url + '?channel=C-1&start=2026-07-01&end=2026-07-21')
        args, kwargs = c.sqlite.query_tel_page.call_args
        self.assertIsNotNone(kwargs.get('start_ts'))
        self.assertIsNotNone(kwargs.get('end_ts'))

    def test_available_channels_in_context(self):
        """The dropdown should list channels from the device tree."""
        self.client.force_login(self.staff)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertIn('C-1', resp.context['available_channels'])

    def test_rows_json_injected_for_chart(self):
        """rows_json carries the decorated rows so ECharts can render."""
        self.client.force_login(self.staff)
        c = _make_mock_container(rows=[
            {'id': 1, 'timestamp': 1700000000.0, 'raw_value': 1.0,
             'predicted_value': None, 'origin': 'acq', 'anomaly_score': None,
             'predicted_anomaly_score': None, 'origin_ts': None,
             'ingested_at': 1700000000, 'deleted_at': None, 'channel': 'C-1'},
        ])
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?channel=C-1')
        rows_json = resp.context['rows_json']
        parsed = json.loads(rows_json)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['id'], 1)
        self.assertEqual(parsed[0]['type_label'], '真实')


class TelemetryViewPaginationTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_telemetry')
        self.staff = User.objects.create_user(
            username='staff2', password='pw', is_staff=True
        )

    def _make_rows(self, n):
        return [
            {'id': i + 1, 'timestamp': 1700000000.0 + i, 'raw_value': float(i),
             'predicted_value': None, 'origin': 'acq', 'anomaly_score': None,
             'predicted_anomaly_score': None, 'origin_ts': None,
             'ingested_at': 1700000000 + i, 'deleted_at': None, 'channel': 'C-1'}
            for i in range(n)
        ]

    def test_total_pages_calculation(self):
        """total_pages = ceil(total_count / limit)."""
        self.client.force_login(self.staff)
        c = _make_mock_container(rows=self._make_rows(20), count=100)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?channel=C-1&limit=20')
        self.assertEqual(resp.context['total_pages'], 5)
        self.assertEqual(resp.context['total_count'], 100)

    def test_page_beyond_last_clamped(self):
        """When page exceeds total_pages, it is clamped to the last page."""
        self.client.force_login(self.staff)
        c = _make_mock_container(rows=self._make_rows(10), count=100)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?channel=C-1&page=99&limit=20')
        self.assertEqual(resp.context['page'], 5)

    def test_pagination_rendered_when_multiple_pages(self):
        self.client.force_login(self.staff)
        c = _make_mock_container(rows=self._make_rows(20), count=100)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?channel=C-1&limit=20')
        self.assertContains(resp, 'phm-pagination')
        self.assertContains(resp, '第 1/5 页')

    def test_page_size_options_in_context(self):
        self.client.force_login(self.staff)
        c = _make_mock_container(rows=self._make_rows(3), count=3)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?channel=C-1')
        self.assertEqual(resp.context['page_size_options'],
                         _TEL_PAGE_SIZE_OPTIONS)
        # The page-size dropdown renders every candidate value.
        for n in _TEL_PAGE_SIZE_OPTIONS:
            self.assertContains(resp, 'value="{}"'.format(n))


# ════════════════════════════════════════════════════════════════════════════
# AJAX endpoint tests
# ════════════════════════════════════════════════════════════════════════════

class TelemetryChannelsApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_telemetry_channels')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_anonymous_redirects(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)

    def test_returns_channels(self):
        self.client.force_login(self.staff)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertIn('C-1', body['channels'])


class TelemetryCreateApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_telemetry_create')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_staff_gets_403(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url, {'channel': 'C-1', 'timestamp': 1700000000,
                       'raw_value': 1.0},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 403)

    def test_superuser_success(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1', 'timestamp': 1700000000,
                           'raw_value': 1.5},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        # insert_tel_manual called with the parsed args.
        args, kwargs = c.sqlite.insert_tel_manual.call_args
        self.assertEqual(kwargs.get('channel'), 'C-1')
        self.assertEqual(kwargs.get('raw_value'), 1.5)

    def test_missing_channel_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'timestamp': 1700000000, 'raw_value': 1.0},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)

    def test_missing_timestamp_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1', 'raw_value': 1.0},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)

    def test_no_value_returns_400(self):
        """At least one value column must be supplied."""
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1', 'timestamp': 1700000000},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)

    def test_iso_timestamp_accepted(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1',
                           'timestamp': '2026-07-21T12:00:00',
                           'raw_value': 1.0},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        args, kwargs = c.sqlite.insert_tel_manual.call_args
        self.assertGreater(kwargs.get('timestamp'), 0)

    def test_insert_failure_returns_500(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        c.sqlite.insert_tel_manual.return_value = False
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1', 'timestamp': 1700000000,
                           'raw_value': 1.0},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 500)


class TelemetryDeleteApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_telemetry_delete')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_staff_gets_403(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url, {'channel': 'C-1', 'rowids': [1, 2]},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 403)

    def test_superuser_success(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        c.sqlite.soft_delete_tel.return_value = 2
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1', 'rowids': [1, 2]},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['status'], 'ok')
        self.assertEqual(resp.json()['deleted'], 2)
        c.sqlite.soft_delete_tel.assert_called_with('C-1', [1, 2])

    def test_missing_channel_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'rowids': [1]},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)

    def test_empty_rowids_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'channel': 'C-1', 'rowids': []},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)


class TelemetryExportApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_telemetry_export')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_missing_channel_returns_400(self):
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 400)

    def test_csv_export_has_bom(self):
        """Export streams CSV with a UTF-8 BOM (Excel-friendly)."""
        self.client.force_login(self.staff)
        rows = [
            {'id': 1, 'timestamp': 1700000000.0, 'raw_value': 1.5,
             'anomaly_score': 0.2, 'predicted_value': None,
             'origin': 'manual', 'origin_ts': None, 'ingested_at': 1700000005,
             'deleted_at': None},
        ]
        c = _make_mock_container()
        c.sqlite.iter_tel_rows.return_value = iter(rows)
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?channel=C-1')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp['Content-Type'])
        body = b''.join(resp.streaming_content)
        # First three bytes = UTF-8 BOM.
        self.assertTrue(body.startswith(b'\xef\xbb\xbf'))
        text = body.decode('utf-8')
        self.assertIn('channel', text)  # header row
        self.assertIn('C-1', text)      # data row
        self.assertIn('manual', text)   # origin column

    def test_export_streams_via_iter_tel_rows(self):
        """Export uses the streaming generator (memory-bounded)."""
        self.client.force_login(self.staff)
        c = _make_mock_container()
        c.sqlite.iter_tel_rows.return_value = iter([])
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url + '?channel=C-1&start=2026-07-01')
            # Drain the streaming response so the generator is actually consumed
            # (StreamingHttpResponse only runs the generator on iteration).
            list(resp.streaming_content)
        args, kwargs = c.sqlite.iter_tel_rows.call_args
        self.assertEqual(args[0], 'C-1')
        self.assertIsNotNone(kwargs.get('start_ts'))
