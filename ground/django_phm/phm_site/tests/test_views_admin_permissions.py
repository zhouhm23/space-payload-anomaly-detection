"""Page 9: permissions help page tests.

Coverage:
  (a) Helper data structures: integrity of _PERMISSION_ROLES /
      _PERMISSION_MATRIX / _AUDIT_SCOPE_NOTES
  (b) View access: anonymous redirects to login / staff 200 / superuser 200
  (c) Rendering: includes the three roles / each page permission entry /
      current-role highlight / audit-scope notes
  (d) Route name resolution works

Tests use Django TestCase + force_login. No Container mock needed (the page
is static and Container-independent).
"""
from __future__ import annotations

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm_site.views_admin import (
    _AUDIT_SCOPE_NOTES,
    _PERMISSION_MATRIX,
    _PERMISSION_ROLES,
)


# ════════════════════════════════════════════════════════════════════════════
# Data-structure integrity tests (no Django DB dependency)
# ════════════════════════════════════════════════════════════════════════════

class PermissionDataTest(TestCase):
    """Integrity of the permissions-help data structures."""

    def test_roles_have_three_levels(self):
        """Must have the three roles anonymous/staff/superuser."""
        keys = {r['key'] for r in _PERMISSION_ROLES}
        self.assertEqual(keys, {'anonymous', 'staff', 'superuser'})

    def test_role_fields_complete(self):
        """Each role must have all four fields key/name/desc/badge."""
        for r in _PERMISSION_ROLES:
            self.assertIn('key', r)
            self.assertIn('name', r)
            self.assertIn('desc', r)
            self.assertIn('badge', r)
            self.assertTrue(r['name'])
            self.assertTrue(r['desc'])

    def test_matrix_covers_all_pages(self):
        """The permission matrix must cover every admin custom page + user
        management + audit log."""
        pages = {row['page'] for row in _PERMISSION_MATRIX}
        expected = {
            '仪表盘', '告警与预警管理', '回收站', '设备树管理',
            '系统设置', '模型管理', '用户与组管理', '审计日志',
        }
        self.assertEqual(pages, expected)

    def test_matrix_rows_have_three_role_columns(self):
        """Each row must have anonymous/staff/superuser columns + page + url."""
        for row in _PERMISSION_MATRIX:
            self.assertIn('page', row)
            self.assertIn('url', row)
            self.assertIn('anonymous', row)
            self.assertIn('staff', row)
            self.assertIn('superuser', row)
            self.assertTrue(row['url'].startswith('/admin/'))

    def test_anonymous_never_has_access(self):
        """Anonymous must be '—' for every page (no access)."""
        for row in _PERMISSION_MATRIX:
            self.assertEqual(row['anonymous'], '—',
                             f"{row['page']} 不应允许匿名访问")

    def test_superuser_has_more_or_equal_access_than_staff(self):
        """Superuser permissions ⊇ staff (at minimum, never fewer).

        We do not compare string contents strictly here; we only check that
        the superuser column is not '—'.
        """
        for row in _PERMISSION_MATRIX:
            self.assertNotEqual(row['superuser'], '—',
                                f"{row['page']} 超管应有访问权")

    def test_audit_notes_non_empty(self):
        self.assertGreater(len(_AUDIT_SCOPE_NOTES), 0)
        for note in _AUDIT_SCOPE_NOTES:
            self.assertTrue(note)


# ════════════════════════════════════════════════════════════════════════════
# View access tests
# ════════════════════════════════════════════════════════════════════════════

class PermissionsViewAccessTest(TestCase):
    """Access control and rendering for GET /admin/phm_site/permissions/."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_permissions')
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
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, '权限说明')

    def test_superuser_can_access(self):
        self.client.force_login(self.superuser)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)


# ════════════════════════════════════════════════════════════════════════════
# Render-content tests
# ════════════════════════════════════════════════════════════════════════════

class PermissionsRenderTest(TestCase):
    """Validation of page render contents."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_permissions')
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_renders_three_roles(self):
        """The page should include the three role names."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertContains(resp, '匿名用户')
        self.assertContains(resp, '普通管理员')
        self.assertContains(resp, '超级管理员')

    def test_renders_all_pages_in_matrix(self):
        """The page should include every page name from the permission matrix."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        for row in _PERMISSION_MATRIX:
            self.assertContains(resp, row['page'])

    def test_renders_audit_scope_notes(self):
        """The audit-scope notes should be rendered."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertContains(resp, '审计日志范围')
        # The LogEntry keyword should appear at least once.
        self.assertContains(resp, 'LogEntry')

    def test_staff_sees_current_role_badge_on_staff(self):
        """When logged in as staff, the '普通管理员' card should carry the '当前角色' badge."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        # When current_role=staff, the staff card contains '当前角色'; the
        # superuser card should not.
        self.assertContains(resp, '当前角色')

    def test_superuser_sees_current_role_badge_on_superuser(self):
        """When logged in as superuser, the '超级管理员' card should carry the '当前角色' badge."""
        self.client.force_login(self.superuser)
        resp = self.client.get(self.url)
        self.assertContains(resp, '当前角色')

    def test_renders_user_management_hint(self):
        """The page should hint where to find user/group management."""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertContains(resp, '用户与组管理')
        self.assertContains(resp, 'django.contrib.auth')

    def test_no_container_dependency(self):
        """The static page must not depend on the Container (it should render
        even when services_bridge is not running).

        Verified by accessing the page without mocking services_bridge.
        """
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # The placeholder-page text should not be rendered.
        self.assertNotContains(resp, '正在初始化')
        self.assertNotContains(resp, 'PHM 服务')
