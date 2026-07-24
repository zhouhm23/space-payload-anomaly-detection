"""Page 1: model-management page tests (now superseded by library_view).

The single-page ``models_view`` was replaced by the 5-sub-menu
``library_view`` (v1.2).  The old ``/admin/phm_site/models/`` route now
returns a 301 permanent redirect to ``/admin/phm_site/library/`` — its
access-control behaviour changed accordingly (every authenticated user
gets a 301, anonymous users still hit the login gate first).

This file retains coverage for the three *helper* functions that
``library_view`` reuses as fallbacks / building blocks:
  - ``_scan_sensor_model_usage`` (legacy @ command substring matcher,
    reused by ``scan_module_usage`` as backward-compat fallback)
  - ``_scan_default_usage`` (kept for callers that still want the legacy
    default-usage shape)
  - ``_check_local_assets`` (reused by ``library_view`` for model cards)

The full library-page coverage (5 tabs, dual panel, read-only, 301) lives
in ``test_views_admin_library.py``.
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


class ModelsRouteRedirectTest(TestCase):
    """The legacy /models/ route now 301-redirects to /library/.

    Detailed library-page coverage (cards, tabs, dual panel) lives in
    test_views_admin_library.py; here we only verify the legacy URL's
    redirect semantics so old bookmarks do not break.
    """

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_models')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_anonymous_also_gets_301(self):
        """Anonymous access → 301 to /library/ (redirect is unauthenticated;
        the auth gate lives on library_view itself).

        Following the redirect lands on /library/ which then 302s to login
        (verified separately in test_views_admin_library.py).
        """
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 301)
        self.assertIn('/admin/phm_site/library/', resp['Location'])
        # Following the chain lands on the login page (library_view is gated).
        resp2 = self.client.get(resp['Location'])
        self.assertEqual(resp2.status_code, 302)
        self.assertIn('/admin/login/', resp2['Location'])

    def test_authenticated_gets_301_to_library(self):
        """Authenticated staff → 301 permanent redirect to /library/."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 301)
        self.assertIn('/admin/phm_site/library/', resp['Location'])


class SensorModelUsageScanTest(TestCase):
    """@ command scanning logic (_scan_sensor_model_usage).

    Retained because ``scan_module_usage`` reuses this helper as its
    backward-compat fallback for sensors whose descriptions still carry a
    legacy ``@tspulse`` / ``@预测模型`` substring (pre-DSL migration).
    """

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
    """Default-usage scanning (_scan_default_usage).

    Retained as a standalone helper (the new ``scan_module_usage`` has its
    own default-flow backfill inside ChannelCalibration, but this legacy
    shape is still consumed by ``models_view``'s old default-usage path).
    """

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
    """Local asset check (_check_local_assets; no torch import).

    Retained because ``library_view`` calls this for every model card to
    surface the local-asset availability badge.
    """

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
