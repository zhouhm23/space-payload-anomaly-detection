"""Page 6: device-tree management tests.

Coverage:
  (a) Pure helpers: _mark_special_sensors / _load_space_channels
  (b) View access: anonymous redirects to login / staff read / superuser
      write / Container-not-ready placeholder page
  (c) AJAX endpoints: save (empty-tree rejection / duplicate sourceId /
      normal save) + channels
"""
from __future__ import annotations

import json
from unittest import mock

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm_site.views_admin import (
    _load_space_channels,
    _mark_special_sensors,
)


# ════════════════════════════════════════════════════════════════════════════
# Helper pure-function tests
# ════════════════════════════════════════════════════════════════════════════

class MarkSpecialSensorsTest(TestCase):

    def test_marks_rul_description_sensor(self):
        tree = [
            {'type': 'sensor', 'name': 'C-1', 'description': 'normal sensor'},
            {'type': 'sensor', 'name': 'C-2', 'description': '@rul:fd001 special'},
        ]
        out = _mark_special_sensors(tree)
        self.assertFalse(out[0]['_special'])
        self.assertTrue(out[1]['_special'])

    def test_marks_isspecial_field(self):
        tree = [
            {'type': 'sensor', 'name': 'X', 'isSpecial': True, 'description': ''},
        ]
        out = _mark_special_sensors(tree)
        self.assertTrue(out[0]['_special'])

    def test_recursive(self):
        tree = [
            {'type': 'folder', 'name': 'f', 'children': [
                {'type': 'sensor', 'name': 'a', 'description': '@rul:fd001'},
                {'type': 'sensor', 'name': 'b', 'description': ''},
            ]},
        ]
        out = _mark_special_sensors(tree)
        self.assertTrue(out[0]['children'][0]['_special'])
        self.assertFalse(out[0]['children'][1]['_special'])

    def test_does_not_mutate_input(self):
        tree = [{'type': 'sensor', 'name': 'a', 'description': '@rul'}]
        _mark_special_sensors(tree)
        self.assertNotIn('_special', tree[0])  # The original object must not be mutated.

    def test_non_list_returns_empty(self):
        self.assertEqual(_mark_special_sensors(None), [])
        self.assertEqual(_mark_special_sensors({}), [])


class LoadSpaceChannelsTest(TestCase):

    def test_reads_existing_file(self):
        """When the real file exists, it should return a non-empty list."""
        channels = _load_space_channels()
        # The file does exist and is non-empty.
        self.assertGreater(len(channels), 0)
        # Each item has a source_id.
        for ch in channels:
            self.assertIn('source_id', ch)
            self.assertIn('label', ch)

    def test_file_missing_returns_empty(self):
        """When the path does not exist, returns []."""
        from django.test import override_settings
        with mock.patch('phm_site.views_admin._SPACE_CHANNELS_PATH',
                        '/nonexistent/path.json'):
            self.assertEqual(_load_space_channels(), [])


# ════════════════════════════════════════════════════════════════════════════
# View access tests
# ════════════════════════════════════════════════════════════════════════════

def _make_mock_container(tree=None, save_result=None):
    c = mock.Mock()
    c.config.load.return_value = {
        'device_tree': tree if tree is not None else [
            {'type': 'folder', 'name': 'f1', 'children': [
                {'type': 'sensor', 'name': 'C-1', 'channelName': 'C-1', 'sourceId': 'file:NASA-MSL/C-1'},
            ]},
        ],
        'aggregation_strategy': 'min',
    }
    c.config.save.return_value = save_result or {'status': 'ok'}
    return c


def _patch_container(c):
    return (
        mock.patch('phm_site.services_bridge.get_container', return_value=c),
        mock.patch('phm_site.services_bridge.get_state', return_value='ready'),
    )


class DeviceTreeViewAccessTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_device_tree')
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
        self.assertContains(resp, '设备树管理')

    def test_container_not_ready_renders_state_page(self):
        self.client.force_login(self.staff)
        with mock.patch('phm_site.services_bridge.get_state', return_value='initializing'):
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_special_sensor_marked_with_star(self):
        """A sensor with an @rul description should have the _special flag injected into the template."""
        self.client.force_login(self.staff)
        c = _make_mock_container(tree=[
            {'type': 'sensor', 'name': 'RUL-1', 'description': '@rul:fd001'},
        ])
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # tree_json should contain _special: true.
        self.assertContains(resp, '_special')


# ════════════════════════════════════════════════════════════════════════════
# AJAX endpoint tests
# ════════════════════════════════════════════════════════════════════════════

class DeviceTreeSaveApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_device_tree_save')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_staff_gets_403(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url, {'device_tree': []},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 403)

    def test_missing_device_tree_returns_400(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container()
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'aggregation_strategy': 'min'},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)

    def test_empty_tree_rejected_by_service(self):
        """Empty tree is rejected by ConfigService.save (returns status=error); the view returns 400."""
        self.client.force_login(self.superuser)
        c = _make_mock_container(save_result={
            'status': 'error', 'message': '拒绝保存空设备树（安全保护）',
            'current_tree': [],
        })
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url, {'device_tree': []},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('安全保护', resp.json()['message'])

    def test_duplicate_source_id_rejected(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container(save_result={
            'status': 'error', 'message': '重复的数据源标识: file:NASA-MSL/C-1',
        })
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url,
                {'device_tree': [
                    {'type': 'sensor', 'sourceId': 'file:NASA-MSL/C-1'},
                    {'type': 'sensor', 'sourceId': 'file:NASA-MSL/C-1'},
                ]},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('重复', resp.json()['message'])

    def test_success(self):
        self.client.force_login(self.superuser)
        c = _make_mock_container(save_result={'status': 'ok'})
        p1, p2 = _patch_container(c)
        with p1, p2:
            resp = self.client.post(
                self.url,
                {'device_tree': [{'type': 'sensor', 'sourceId': 'x'}]},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['status'], 'ok')

    def test_invalid_json_returns_400(self):
        self.client.force_login(self.superuser)
        resp = self.client.post(
            self.url, 'not json', content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_non_dict_body_returns_400(self):
        self.client.force_login(self.superuser)
        resp = self.client.post(
            self.url, '[1,2,3]', content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)


class DeviceTreeChannelsApiTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_device_tree_channels')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_staff_can_access(self):
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertIn('channels', body)

    def test_anonymous_redirects(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)


# ════════════════════════════════════════════════════════════════════════════
# @command DSL save hook (Plan 3): validation blocking + persistence
# ════════════════════════════════════════════════════════════════════════════

class DeviceTreeSaveDslHookTest(TestCase):
    """The save API must block on hard DSL errors and persist valid configs.

    Coverage:
      - E1-E5 each block the save with HTTP 400 + structured error payload.
      - Sensors with no description / no @commands pass through unchanged
        (backward compatibility for existing deployments).
      - A valid @command description is persisted to channel_calibration.json
        via CalibrationConfig.upsert.
      - The live validation API returns errors/warnings as JSON.
    """

    def setUp(self):
        self.client = Client()
        self.save_url = reverse('phm_admin_device_tree_save')
        self.validate_url = reverse('phm_admin_device_tree_validate_dsl')
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def _post_save(self, tree):
        self.client.force_login(self.superuser)
        c = _make_mock_container(save_result={'status': 'ok'})
        p1, p2 = _patch_container(c)
        with p1, p2:
            return self.client.post(
                self.save_url, json.dumps({'device_tree': tree}),
                content_type='application/json',
            )

    # ── Backward compatibility ─────────────────────────────────────────

    def test_sensor_without_description_passes(self):
        """A sensor with no description must not be blocked by the DSL hook."""
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1'},
        ])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['status'], 'ok')

    def test_sensor_with_prose_only_description_passes(self):
        """Plain prose (no @commands) must not be blocked."""
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '普通温度传感器'},
        ])
        self.assertEqual(resp.status_code, 200)

    def test_legacy_at_rul_description_passes(self):
        """Legacy ``@rul:xxx`` is unknown to the DSL → treated as prose → passes."""
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '@rul:fd001 special'},
        ])
        self.assertEqual(resp.status_code, 200)

    # ── E1-E5 hard errors block the save ───────────────────────────────

    def test_e1_unknown_algorithm_blocks_save(self):
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '@算法=does_not_exist'},
        ])
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertEqual(body['channel'], 'X-1')
        self.assertTrue(any('E1' in e for e in body['errors']))

    def test_e2_skip_model_mutex_blocks_save(self):
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '@算法=tspulse @跳过模型'},
        ])
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(any('E2' in e for e in resp.json()['errors']))

    def test_e3_setpoint_no_anchor_blocks_save(self):
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '@算法=l1_setpoint'},
        ])
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(any('E3' in e for e in resp.json()['errors']))

    def test_e4_threshold_out_of_range_blocks_save(self):
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '@算法=tspulse @阈值=1.5'},
        ])
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(any('E4' in e for e in resp.json()['errors']))

    def test_e5_param_for_undeclared_module_blocks_save(self):
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '@算法=tspulse @参数.l1_sigma.sigma_k=4'},
        ])
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(any('E5' in e for e in resp.json()['errors']))

    # ── Error payload structure ────────────────────────────────────────

    def test_error_payload_carries_channel_and_description(self):
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '@算法=bad'},
        ])
        body = resp.json()
        self.assertEqual(body['status'], 'error')
        self.assertEqual(body['channel'], 'X-1')
        self.assertIn('description', body)
        self.assertEqual(body['description'], '@算法=bad')

    def test_first_failing_sensor_reported(self):
        """When multiple sensors fail, only the first is reported (fix-and-resubmit)."""
        resp = self._post_save([
            {'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
             'description': '@算法=bad1'},
            {'type': 'sensor', 'channelName': 'X-2', 'sourceId': 'X-2',
             'description': '@算法=bad2'},
        ])
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()['channel'], 'X-1')

    # ── Persistence (best-effort, mocked) ──────────────────────────────

    def test_valid_dsl_persists_calibration(self):
        """A valid @command description triggers CalibrationConfig.upsert."""
        with mock.patch('phm_site.views_admin.CalibrationConfig') as MockCC:
            instance = MockCC.return_value
            instance.get.return_value = None
            c = _make_mock_container(save_result={'status': 'ok'})
            p1, p2 = _patch_container(c)
            with p1, p2:
                resp = self._post_save([{
                    'type': 'sensor', 'channelName': 'X-1', 'sourceId': 'X-1',
                    'description': '@算法=tspulse @阈值=0.6',
                }])
        self.assertEqual(resp.status_code, 200)
        # upsert must have been called once for X-1.
        instance.upsert.assert_called_once()
        args = instance.upsert.call_args[0]
        self.assertEqual(args[0], 'X-1')

    def test_persistence_failure_does_not_break_save(self):
        """If CalibrationConfig.upsert raises, the save itself still succeeds."""
        with mock.patch('phm_site.views_admin.CalibrationConfig') as MockCC:
            instance = MockCC.return_value
            instance.get.return_value = None
            instance.upsert.side_effect = RuntimeError('disk full')
            c = _make_mock_container(save_result={'status': 'ok'})
            p1, p2 = _patch_container(c)
            with p1, p2:
                resp = self._post_save([{
                    'type': 'sensor', 'sourceId': 'X-1',
                    'description': '@算法=tspulse',
                }])
        # Tree was already saved; calibration write failure is logged, not raised.
        self.assertEqual(resp.status_code, 200)


class DeviceTreeValidateDslApiTest(TestCase):
    """The live validation API (POST description → errors/warnings)."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_device_tree_validate_dsl')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_valid_description_returns_no_errors(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url,
            json.dumps({'description': '@算法=tspulse'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['errors'], [])
        self.assertTrue(body['has_commands'])

    def test_invalid_description_returns_errors(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url,
            json.dumps({'description': '@算法=bad @阈值=2'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        # Both E1 and E4 should be reported.
        self.assertTrue(any('E1' in e for e in body['errors']))
        self.assertTrue(any('E4' in e for e in body['errors']))

    def test_empty_description_returns_no_errors_but_has_commands_false(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url,
            json.dumps({'description': 'just prose'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['errors'], [])
        self.assertFalse(body['has_commands'])

    def test_anonymous_redirects(self):
        resp = self.client.post(
            self.url,
            json.dumps({'description': '@算法=tspulse'}),
            content_type='application/json',
        )
        # staff_member_required redirects non-logged-in users.
        self.assertEqual(resp.status_code, 302)

    def test_invalid_json_returns_400(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url, 'not json', content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)
