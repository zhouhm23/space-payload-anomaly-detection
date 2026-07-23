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
