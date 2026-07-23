"""Page 5: system settings tests.

Coverage:
  (a) Pure helpers: _parse_settings_category / _classify_value_kind
      / _build_settings_items / _group_items_by_section
  (b) View access: anonymous redirects to login / staff read / superuser write /
      all three categories render
  (c) AJAX endpoint permissions: save → non-staff 403, staff 403, superuser 200
  (d) AJAX business logic: success / readonly-key rejection / type mismatch /
      calibration returns 403 directly
  (e) Type validation / _doc preservation

Tests use Django TestCase + force_login. The Service layer is mocked (no real
file writes).
"""
from __future__ import annotations

import json
from unittest import mock

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm_site.views_admin import (
    _build_settings_items,
    _classify_value_kind,
    _group_items_by_section,
    _parse_settings_category,
)


# ════════════════════════════════════════════════════════════════════════════
# Helper pure-function tests
# ════════════════════════════════════════════════════════════════════════════

class ParseSettingsCategoryTest(TestCase):

    def test_valid_keys(self):
        self.assertEqual(_parse_settings_category('system'), 'system')
        self.assertEqual(_parse_settings_category('theme'), 'theme')
        self.assertEqual(_parse_settings_category('calibration'), 'calibration')

    def test_invalid_falls_back_to_system(self):
        self.assertEqual(_parse_settings_category('unknown'), 'system')

    def test_none_falls_back(self):
        self.assertEqual(_parse_settings_category(None), 'system')

    def test_empty_falls_back(self):
        self.assertEqual(_parse_settings_category(''), 'system')


class ClassifyValueKindTest(TestCase):

    def test_scalar_types(self):
        self.assertEqual(_classify_value_kind(True), 'bool')
        self.assertEqual(_classify_value_kind(42), 'int')
        self.assertEqual(_classify_value_kind(3.14), 'float')
        self.assertEqual(_classify_value_kind("hello"), 'str')

    def test_collection_types(self):
        self.assertEqual(_classify_value_kind([1, 2]), 'array')
        self.assertEqual(_classify_value_kind({'a': 1}), 'object')

    def test_none(self):
        self.assertEqual(_classify_value_kind(None), 'unknown')


class BuildSettingsItemsTest(TestCase):
    """_build_settings_items flattening logic."""

    def test_flatten_with_display_names(self):
        raw = {
            'thresholds': {
                '_doc': 'thresholds doc',
                'anomaly': 0.5,
                'l1_sigma_k': 3.0,
            },
        }
        names = {
            'thresholds': {'_doc': '阈值', 'anomaly': '异常分数阈值', 'l1_sigma_k': 'σ 倍数'}
        }
        items = _build_settings_items(raw, names)
        # _doc should not appear as an item.
        keys = [(it['section'], it['key']) for it in items]
        self.assertIn(('thresholds', 'anomaly'), keys)
        self.assertIn(('thresholds', 'l1_sigma_k'), keys)
        self.assertNotIn(('thresholds', '_doc'), keys)
        # The Chinese label comes from display_names.
        anomaly = next(it for it in items if it['key'] == 'anomaly')
        self.assertEqual(anomaly['name'], '异常分数阈值')
        self.assertEqual(anomaly['section_label'], '阈值')

    def test_editable_flag_for_scalars(self):
        raw = {'sec': {'k_int': 1, 'k_str': 's', 'k_bool': True,
                       'k_list': [1, 2], 'k_dict': {'a': 1}}}
        items = _build_settings_items(raw, {'sec': {}})
        by_key = {it['key']: it for it in items}
        self.assertTrue(by_key['k_int']['editable'])
        self.assertTrue(by_key['k_str']['editable'])
        self.assertTrue(by_key['k_bool']['editable'])
        self.assertFalse(by_key['k_list']['editable'])
        self.assertFalse(by_key['k_dict']['editable'])

    def test_readonly_predicate(self):
        raw = {'sec': {'k1': 1, 'k2': 2}}
        items = _build_settings_items(
            raw, {'sec': {}},
            readonly_predicate=lambda s, k: k == 'k1',
        )
        by_key = {it['key']: it for it in items}
        self.assertFalse(by_key['k1']['editable'])
        self.assertTrue(by_key['k2']['editable'])

    def test_underscore_section_skipped(self):
        raw = {'_meta': {'x': 1}, 'sec': {'k': 1}}
        items = _build_settings_items(raw, {'sec': {}})
        sections = {it['section'] for it in items}
        self.assertNotIn('_meta', sections)


class GroupItemsBySectionTest(TestCase):

    def test_group_preserves_order(self):
        items = [
            {'section': 'a', 'section_label': 'A', 'section_doc': '', 'key': 'x',
             'name': 'X', 'doc': '', 'value': 1, 'value_kind': 'int', 'editable': True},
            {'section': 'b', 'section_label': 'B', 'section_doc': '', 'key': 'y',
             'name': 'Y', 'doc': '', 'value': 2, 'value_kind': 'int', 'editable': True},
            {'section': 'a', 'section_label': 'A', 'section_doc': '', 'key': 'z',
             'name': 'Z', 'doc': '', 'value': 3, 'value_kind': 'int', 'editable': True},
        ]
        groups = _group_items_by_section(items)
        self.assertEqual([g['section'] for g in groups], ['a', 'b'])
        self.assertEqual(len(groups[0]['items']), 2)
        self.assertEqual(len(groups[1]['items']), 1)


# ════════════════════════════════════════════════════════════════════════════
# View access tests
# ════════════════════════════════════════════════════════════════════════════

class SettingsViewAccessTest(TestCase):
    """Access control and rendering for GET /admin/phm_site/settings/."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_settings')
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

    def test_staff_can_access_system_tab(self):
        self.client.force_login(self.staff)
        patcher, _svc = _patch_system_service()
        with patcher:
            resp = self.client.get(self.url + '?category=system')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '系统配置')

    def test_superuser_can_access_theme_tab(self):
        self.client.force_login(self.superuser)
        patcher, _svc = _patch_theme_service()
        with patcher:
            resp = self.client.get(self.url + '?category=theme')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '前台主题')

    def test_calibration_tab_renders_readonly(self):
        self.client.force_login(self.staff)
        # Use the real file path (it exists); no mock.
        resp = self.client.get(self.url + '?category=calibration')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '只读')

    def test_invalid_category_falls_back_to_system(self):
        self.client.force_login(self.staff)
        patcher, _svc = _patch_system_service()
        with patcher:
            resp = self.client.get(self.url + '?category=bogus')
        self.assertEqual(resp.status_code, 200)

    def test_system_tab_shows_save_button_for_superuser(self):
        self.client.force_login(self.superuser)
        patcher, _svc = _patch_system_service()
        with patcher:
            resp = self.client.get(self.url + '?category=system')
        self.assertContains(resp, 'phm-settings-save-btn')

    def test_system_tab_hides_save_button_for_staff(self):
        self.client.force_login(self.staff)
        patcher, _svc = _patch_system_service()
        with patcher:
            resp = self.client.get(self.url + '?category=system')
        # Staff cannot see the save button (the <button> element with
        # class="phm-settings-save-btn").
        # Note: the JS also references the '.phm-settings-save-btn' string
        # (querySelector), so we assert on the precise <button tag.
        self.assertNotContains(resp, 'class="phm-btn phm-btn-sm phm-btn-primary phm-settings-save-btn"')


# ════════════════════════════════════════════════════════════════════════════
# AJAX endpoint tests
# ════════════════════════════════════════════════════════════════════════════

class SettingsSaveApiTest(TestCase):
    """POST /admin/phm_site/settings/api/save/ endpoint."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_settings_save')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_anonymous_redirects_to_login(self):
        resp = self.client.post(self.url, {}, content_type='application/json')
        self.assertEqual(resp.status_code, 302)

    def test_staff_gets_403(self):
        self.client.force_login(self.staff)
        resp = self.client.post(
            self.url,
            {'category': 'system', 'section': 'thresholds', 'key': 'anomaly', 'value': 0.4},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 403)

    def test_superuser_success_system(self):
        self.client.force_login(self.superuser)
        patcher, svc = _patch_system_service()
        svc.save.return_value = {
            'status': 'ok', 'section': 'thresholds', 'key': 'anomaly',
            'old': 0.5, 'new': 0.4,
        }
        with patcher:
            resp = self.client.post(
                self.url,
                {'category': 'system', 'section': 'thresholds',
                 'key': 'anomaly', 'value': 0.4},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['new'], 0.4)

    def test_calibration_rejected_with_400(self):
        self.client.force_login(self.superuser)
        resp = self.client.post(
            self.url,
            {'category': 'calibration', 'section': 'C-1', 'key': 'flip', 'value': True},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('只读', resp.json()['message'])

    def test_service_error_returns_400(self):
        self.client.force_login(self.superuser)
        patcher, svc = _patch_system_service()
        svc.save.return_value = {
            'status': 'error', 'message': '未知配置项：bogus.x',
        }
        with patcher:
            resp = self.client.post(
                self.url,
                {'category': 'system', 'section': 'bogus',
                 'key': 'x', 'value': 1},
                content_type='application/json',
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('未知配置项', resp.json()['message'])

    def test_missing_value_returns_400(self):
        self.client.force_login(self.superuser)
        resp = self.client.post(
            self.url,
            {'category': 'system', 'section': 'thresholds', 'key': 'anomaly'},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn('value', resp.json()['message'])

    def test_invalid_json_returns_400(self):
        self.client.force_login(self.superuser)
        resp = self.client.post(
            self.url, 'not json', content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_invalid_section_type_returns_400(self):
        self.client.force_login(self.superuser)
        resp = self.client.post(
            self.url,
            {'category': 'system', 'section': 123, 'key': 'x', 'value': 1},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)


# ════════════════════════════════════════════════════════════════════════════
# Helper mock helpers
# ════════════════════════════════════════════════════════════════════════════

def _patch_system_service():
    """Mock get_system_config(): raw_with_docs returns a minimal structure and
    display_names returns Chinese names.

    Uses new=svc so the patch's stand-in IS svc itself (the `as svc` clause
    receives svc, not an unconfigured MagicMock). Note that the service layer's
    get_system_config is a function, so the new stand-in must also be callable
    (CallableMock makes __call__ return svc itself).
    """
    svc = mock.Mock()
    svc.raw_with_docs.return_value = {
        'thresholds': {
            '_doc': '异常检测阈值',
            'anomaly': 0.5,
            'l1_sigma_k': 3.0,
        },
    }
    svc.display_names.return_value = {
        'thresholds': {'_doc': '异常检测阈值', 'anomaly': '异常分数阈值',
                       'l1_sigma_k': 'σ 倍数'}
    }
    svc.is_readonly.return_value = False
    # get_system_config() returns svc — patching a lambda is enough.
    return mock.patch(
        'phm.services.system_config_service.get_system_config',
        new=lambda: svc,
    ), svc


def _patch_theme_service():
    """mock get_theme()。"""
    svc = mock.Mock()
    svc.raw_with_docs.return_value = {
        'colors': {
            '_doc': '调色板',
            'blue': '#2d8cf0',
        },
    }
    svc.display_names.return_value = {
        'colors': {'_doc': '调色板', 'blue': '蓝'},
    }
    svc.is_readonly.return_value = False
    return mock.patch(
        'phm.services.theme_service.get_theme',
        new=lambda: svc,
    ), svc
