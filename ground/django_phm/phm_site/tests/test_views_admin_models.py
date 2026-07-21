"""第 1 页：模型管理页测试。

覆盖：
  (a) 匿名访问 302 跳登录
  (b) staff 访问 200 + 关键内容渲染
  (c) Container 未就绪时渲染状态占位页（不 500）
  (d) 设备树 @ 命令扫描正确（_scan_sensor_model_usage）
  (e) 默认使用情况扫描（_scan_default_usage）
  (f) 本地资产检查（_check_local_assets，不 import torch）

测试用 Django TestCase（pytest-django 也能跑），不依赖真实 Container 状态——
models_view 的设备树读取有 try/except 兜底，Container 未就绪时走占位页。
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
    """页面访问权限与渲染。"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('phm_admin_models')
        # staff 用户（非超管）
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.superuser = User.objects.create_superuser(
            username='admin1', password='pw', email='a@b.c'
        )

    def test_anonymous_redirects_to_login(self):
        """匿名访问 302 跳登录页（需求书"没登录显示登录页"）。"""
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/admin/login/', resp['Location'])

    def test_staff_can_access(self):
        """staff 用户可访问（模型管理是只读页，所有 staff 可看）。"""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_superuser_can_access(self):
        self.client.force_login(self.superuser)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_renders_model_cards(self):
        """页面渲染出 MODEL_REGISTRY 中所有模型的卡片。"""
        self.client.force_login(self.staff)
        resp = self.client.get(self.url)
        self.assertContains(resp, '模型管理')
        for key in MODEL_REGISTRY:
            self.assertContains(resp, key)
        # 资产状态徽章二选一
        self.assertTrue(
            b'\xe8\xb5\x84\xe4\xba\xa7\xe5\xb0\xb1\xe7\xbb\xaa' in resp.content  # "资产就绪"
            or b'\xe8\xb5\x84\xe4\xba\xa7\xe7\xbc\xba\xe5\xa4\xb1' in resp.content  # "资产缺失"
        )

    def test_container_not_ready_still_renders(self):
        """模型管理页不依赖 Container：即使 PHM 未就绪也能渲染（只读 MODEL_REGISTRY）。

        这是产品设计——模型信息是静态元数据 + 本地资产存在性检查，
        不需要加载 torch / 不需要 Container。设备树扫描有 try/except 兜底。
        """
        from unittest import mock
        self.client.force_login(self.staff)
        with mock.patch('phm_site.views_admin.services_bridge') as mock_sb:
            mock_sb.get_state.return_value = 'initializing'
            mock_sb.get_init_error.return_value = None
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # 仍渲染模型卡片（不 500，不走占位页）
        for key in MODEL_REGISTRY:
            self.assertIn(key, resp.content.decode('utf-8'))


class SensorModelUsageScanTest(TestCase):
    """@ 命令扫描逻辑（_scan_sensor_model_usage）。"""

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
        """@异常检测模型 应映射到 tspulse。"""
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
        """同一传感器描述里 @tspulse 出现多次只计一次。"""
        tree = [
            {'type': 'sensor', 'name': 'D', 'description': '@tspulse @tspulse'},
        ]
        usage = _scan_sensor_model_usage(tree)
        self.assertEqual(usage.get('tspulse'), ['D'])

    def test_empty_tree(self):
        self.assertEqual(_scan_sensor_model_usage([]), {})
        self.assertEqual(_scan_sensor_model_usage(None), {})


class DefaultUsageScanTest(TestCase):
    """默认使用情况扫描（_scan_default_usage）。"""

    def test_normal_sensor_defaults_to_tspulse_and_ttm(self):
        """普通传感器（无 @ 命令）默认用 tspulse + ttm_r3。"""
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
        """有显式 @ 命令的传感器不计入默认使用。"""
        tree = [
            {'type': 'sensor', 'name': 'E1', 'description': '@tspulse'},
        ]
        usage = _scan_default_usage(tree)
        self.assertNotIn('E1', usage['tspulse'])


class CheckLocalAssetsTest(TestCase):
    """本地资产检查（_check_local_assets，不 import torch）。"""

    def test_unknown_model_key(self):
        result = _check_local_assets('nonexistent_key')
        self.assertFalse(result['available'])
        self.assertIn('未知', result['note'])

    def test_known_model_returns_structure(self):
        """每个 registry key 都应返回完整结构字段。"""
        for key in MODEL_REGISTRY:
            result = _check_local_assets(key)
            self.assertIn('available', result)
            self.assertIn('path', result)
            self.assertIn('note', result)
            self.assertIsInstance(result['available'], bool)

    def test_rul_local_weights_check(self):
        """RUL 走本地权重路径检查（不 import torch）。"""
        result = _check_local_assets('rul')
        # 不断言 available 值（依赖运行环境），只断言走对了分支
        self.assertIn('note', result)
        # 路径应指向 models/rul/
        self.assertIn('rul', result['path'].lower())
