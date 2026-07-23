"""Page 1: model-management page tests.

Coverage:
  (a) Anonymous access → 302 redirect to login
  (b) Staff access → 200 + key content rendered
  (c) Container-not-ready → renders the status placeholder page (no 500)
  (d) Device-tree @ command scanning is correct (_scan_sensor_model_usage)
  (e) Default usage scanning (_scan_default_usage)
  (f) Local asset check (_check_local_assets; no torch import)

Tests use Django TestCase (pytest-django also works) and do not depend on the
real Container state — models_view's device-tree read has a try/except
fallback that renders the placeholder page when the Container is not ready.
"""
from __future__ import annotations

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm_site.views_admin import (
    _check_local_assets,
    _scan_default_usage,
    _scan_sensor_model_usage,
)
from phm.algorithm._registry import MODEL_REGISTRY


class ModelsViewAccessTest(TestCase):
    """Page access control and rendering."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_models')
        # Staff user (not superuser).
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_anonymous_redirects_to_login(self):
        """Anonymous access → 302 redirect to the login page (spec: "show login page when not logged in")."""
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/admin/login/', resp['Location'])

    def test_staff_can_access(self):
        """Staff users can access (model management is read-only; all staff can view)."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_superuser_can_access(self):
        self.client.force_login(self.superuser)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_renders_model_cards(self):
        """The page should render a card for every model in MODEL_REGISTRY."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertContains(resp, '模型管理')
        for key in MODEL_REGISTRY:
            self.assertContains(resp, key)
        # Exactly one of the asset-status badges is present.
        self.assertTrue(
            b'\xe8\xb5\x84\xe4\xba\xa7\xe5\xb0\xb1\xe7\xbb\xaa' in resp.content  # "资产就绪"
            or b'\xe8\xb5\x84\xe4\xba\xa7\xe7\xbc\xba\xe5\xa4\xb1' in resp.content  # "资产缺失"
        )

    def test_deploy_label_in_cards(self):
        """Each card renders a deploy label (ground / space)."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        # The 3 default models are all ground → at least 3 "ground" badges.
        self.assertContains(resp, '地基')
        # Each card dict carries deploy / deploy_label fields.
        for card in resp.context['cards']:
            self.assertIn('deploy', card)
            self.assertIn('deploy_label', card)
            entry = MODEL_REGISTRY[card['key']]
            self.assertEqual(card['deploy'], entry.deploy)

    def test_container_not_ready_still_renders(self):
        """The model-management page does not depend on the Container: it
        renders even when PHM is not ready (reads MODEL_REGISTRY only).

        This is by design — model info is static metadata + a local asset
        existence check; no torch load / no Container needed. The device-tree
        scan has a try/except fallback.
        """
        from unittest import mock
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as mock_sb:
            mock_sb.get_state.return_value = 'initializing'
            mock_sb.get_init_error.return_value = None
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # Model cards are still rendered (no 500, no placeholder page).
        for key in MODEL_REGISTRY:
            self.assertIn(key, resp.content.decode('utf-8'))


class SensorModelUsageScanTest(TestCase):
    """@ command scanning logic (_scan_sensor_model_usage)."""

    def test_explicit_at_command_detected(self):
        tree = [
            {'type': 'sensor', 'name': 'S1', 'description': '载荷电流 @tspulse 监测'},
            {'type': 'sensor', 'name': 'S2', 'description': '预测通道 @预测模型'},
            {'type': 'sensor', 'name': 'R1', 'description': '退化 @rul:fd001', 'isSpecial': True},
        ]
        usage = _scan_sensor_model_usage(tree)
        self.assertEqual(usage.get('tspulse'), ['S1'])
        self.assertEqual(usage.get('ttm_r3'), ['S2'])
        self.assertEqual(usage.get('rul'), ['R1'])

    def test_chinese_at_command_alias(self):
        """@异常检测模型 should map to tspulse."""
        tree = [
            {'type': 'sensor', 'name': 'X1', 'description': '@异常检测模型 备用'},
        ]
        usage = _scan_sensor_model_usage(tree)
        self.assertEqual(usage.get('tspulse'), ['X1'])

    def test_no_at_command_empty_usage(self):
        tree = [
            {'type': 'sensor', 'name': 'S1', 'description': '普通传感器无 @ 命令'},
        ]
        usage = _scan_sensor_model_usage(tree)
        self.assertEqual(usage, {})

    def test_nested_folder_scanned(self):
        tree = [
            {'type': 'folder', 'name': 'F1', 'children': [
                {'type': 'sensor', 'name': 'N1', 'description': '@tspulse'},
            ]},
        ]
        usage = _scan_sensor_model_usage(tree)
        self.assertEqual(usage.get('tspulse'), ['N1'])

    def test_duplicate_sensor_dedup(self):
        """When @tspulse appears multiple times in one sensor's description, it is counted once."""
        tree = [
            {'type': 'sensor', 'name': 'D', 'description': '@tspulse @tspulse'},
        ]
        usage = _scan_sensor_model_usage(tree)
        self.assertEqual(usage.get('tspulse'), ['D'])

    def test_empty_tree(self):
        self.assertEqual(_scan_sensor_model_usage([]), {})
        self.assertEqual(_scan_sensor_model_usage(None), {})


class DefaultUsageScanTest(TestCase):
    """Default-usage scanning (_scan_default_usage)."""

    def test_normal_sensor_defaults_to_tspulse_and_ttm(self):
        """A normal sensor (no @ command) defaults to tspulse + ttm_r3."""
        tree = [
            {'type': 'sensor', 'name': 'N1', 'description': '普通通道'},
        ]
        usage = _scan_default_usage(tree)
        self.assertIn('N1', usage['tspulse'])
        self.assertIn('N1', usage['ttm_r3'])
        self.assertEqual(usage['rul'], [])

    def test_special_sensor_defaults_to_rul(self):
        tree = [
            {'type': 'sensor', 'name': 'R1', 'isSpecial': True, 'description': ''},
        ]
        usage = _scan_default_usage(tree)
        self.assertIn('R1', usage['rul'])
        self.assertEqual(usage['tspulse'], [])

    def test_explicit_at_not_counted_as_default(self):
        """A sensor with an explicit @ command is not counted as a default usage."""
        tree = [
            {'type': 'sensor', 'name': 'E1', 'description': '@tspulse'},
        ]
        usage = _scan_default_usage(tree)
        self.assertNotIn('E1', usage['tspulse'])


class CheckLocalAssetsTest(TestCase):
    """Local asset check (_check_local_assets; no torch import)."""

    def test_unknown_model_key(self):
        result = _check_local_assets('nonexistent_key')
        self.assertFalse(result['available'])
        self.assertIn('未知', result['note'])

    def test_known_model_returns_structure(self):
        """Every registry key should return the complete set of structural fields."""
        for key in MODEL_REGISTRY:
            result = _check_local_assets(key)
            self.assertIn('available', result)
            self.assertIn('path', result)
            self.assertIn('note', result)
            self.assertIsInstance(result['available'], bool)

    def test_rul_local_weights_check(self):
        """RUL takes the local-weights path check (no torch import)."""
        result = _check_local_assets('rul')
        # Do not assert the available value (environment-dependent); only assert
        # the right branch was taken.
        self.assertIn('note', result)
        # The path should point to models/rul/.
        self.assertIn('rul', result['path'].lower())
