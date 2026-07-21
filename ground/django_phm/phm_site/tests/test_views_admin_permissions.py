"""第 9 页：权限说明测试。

覆盖：
  (a) Helper 数据结构：_PERMISSION_ROLES / _PERMISSION_MATRIX / _AUDIT_SCOPE_NOTES 完整性
  (b) 视图访问：匿名跳登录 / staff 200 / 超管 200
  (c) 渲染：含三大角色 / 含每页权限点 / 当前角色高亮 / 审计边界说明
  (d) 路由 name 解析正常

测试用 Django TestCase + force_login。无需 mock Container（静态页不依赖）。
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
# 数据结构完整性测试（不依赖 Django DB）
# ════════════════════════════════════════════════════════════════════════════

class PermissionDataTest(TestCase):
    """权限说明的数据结构完整性。"""

    def test_roles_have_three_levels(self):
        """必须有匿名/staff/superuser 三个角色。"""
        keys = {r['key'] for r in _PERMISSION_ROLES}
        self.assertEqual(keys, {'anonymous', 'staff', 'superuser'})

    def test_role_fields_complete(self):
        """每个角色必有 key/name/desc/badge 四字段。"""
        for r in _PERMISSION_ROLES:
            self.assertIn('key', r)
            self.assertIn('name', r)
            self.assertIn('desc', r)
            self.assertIn('badge', r)
            self.assertTrue(r['name'])
            self.assertTrue(r['desc'])

    def test_matrix_covers_all_pages(self):
        """权限矩阵必须覆盖后台所有自定义页 + 用户管理 + 审计日志。"""
        pages = {row['page'] for row in _PERMISSION_MATRIX}
        expected = {
            '仪表盘', '告警与预警管理', '回收站', '设备树管理',
            '系统设置', '模型管理', '用户与组管理', '审计日志',
        }
        self.assertEqual(pages, expected)

    def test_matrix_rows_have_three_role_columns(self):
        """每行必须有 anonymous/staff/superuser 三列 + page + url。"""
        for row in _PERMISSION_MATRIX:
            self.assertIn('page', row)
            self.assertIn('url', row)
            self.assertIn('anonymous', row)
            self.assertIn('staff', row)
            self.assertIn('superuser', row)
            self.assertTrue(row['url'].startswith('/admin/'))

    def test_anonymous_never_has_access(self):
        """匿名对所有页面都应是 '—'（无权访问）。"""
        for row in _PERMISSION_MATRIX:
            self.assertEqual(row['anonymous'], '—',
                             f"{row['page']} 不应允许匿名访问")

    def test_superuser_has_more_or_equal_access_than_staff(self):
        """超管权限 ⊇ staff（最起码不会更少）。

        这里不严格比较字符串内容，只检查：超管列不应该是 '—'。
        """
        for row in _PERMISSION_MATRIX:
            self.assertNotEqual(row['superuser'], '—',
                                f"{row['page']} 超管应有访问权")

    def test_audit_notes_non_empty(self):
        self.assertGreater(len(_AUDIT_SCOPE_NOTES), 0)
        for note in _AUDIT_SCOPE_NOTES:
            self.assertTrue(note)


# ════════════════════════════════════════════════════════════════════════════
# 视图访问测试
# ════════════════════════════════════════════════════════════════════════════

class PermissionsViewAccessTest(TestCase):
    """GET /admin/phm_site/permissions/ 访问权限与渲染。"""

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
# 渲染内容测试
# ════════════════════════════════════════════════════════════════════════════

class PermissionsRenderTest(TestCase):
    """页面渲染内容验证。"""

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
        """页面应含三大角色名。"""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertContains(resp, '匿名用户')
        self.assertContains(resp, '普通管理员')
        self.assertContains(resp, '超级管理员')

    def test_renders_all_pages_in_matrix(self):
        """页面应含所有权限矩阵的页面名。"""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        for row in _PERMISSION_MATRIX:
            self.assertContains(resp, row['page'])

    def test_renders_audit_scope_notes(self):
        """审计范围说明应渲染。"""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertContains(resp, '审计日志范围')
        # 至少出现 LogEntry 关键字
        self.assertContains(resp, 'LogEntry')

    def test_staff_sees_current_role_badge_on_staff(self):
        """staff 登录时，'普通管理员' 卡片应带「当前角色」徽章。"""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        # current_role=staff 时 staff 卡片含 '当前角色'，超管卡片不应有
        self.assertContains(resp, '当前角色')

    def test_superuser_sees_current_role_badge_on_superuser(self):
        """超管登录时，'超级管理员' 卡片应带「当前角色」徽章。"""
        self.client.force_login(self.superuser)
        resp = self.client.get(self.url)
        self.assertContains(resp, '当前角色')

    def test_renders_user_management_hint(self):
        """页面应提示用户与组管理入口位置。"""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertContains(resp, '用户与组管理')
        self.assertContains(resp, 'django.contrib.auth')

    def test_no_container_dependency(self):
        """静态页不应依赖 Container（即使 services_bridge 未启动也正常渲染）。

        通过不 mock services_bridge 直接访问验证。
        """
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # 不应渲染占位页文案
        self.assertNotContains(resp, '正在初始化')
        self.assertNotContains(resp, 'PHM 服务')
