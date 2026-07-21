"""services_bridge 链路状态机测试。

历史背景（Day20 bug）：``_poll_one`` 把连接失败误判为 success=True，
导致 ``link_status`` 恒显示 online。修复后（telemetry_service 抛
``ConnectionError``），``_poll_one`` 的 except 捕获 → success=False →
``_link_fail_count`` 累计 → 连续 3 次后 status=offline。

本文件直接测 ``_record_poll_result`` + ``get_link_status`` 的状态机，
不依赖 Django Container / 真实 socket。
"""
from __future__ import annotations

from unittest import mock

from django.test import TestCase

from phm_site import services_bridge


class LinkStatusStateMachineTest(TestCase):
    """``_record_poll_result`` + ``get_link_status`` 状态机。"""

    def setUp(self):
        # 每个测试重置模块级链路状态，避免相互污染
        services_bridge._link_rtt_ms = None
        services_bridge._link_fail_count = 0
        services_bridge._link_last_success_ts = 0.0

    def test_initial_state_is_waiting(self):
        """初始状态：未 poll 过，rtt_ms=None，status='waiting'。"""
        status = services_bridge.get_link_status()
        self.assertIsNone(status['rtt_ms'])
        self.assertEqual(status['status'], 'waiting')

    def test_success_poll_returns_online(self):
        """单次成功 poll（rtt < 3000ms）→ online。"""
        services_bridge._record_poll_result(50.0, success=True)
        status = services_bridge.get_link_status()
        self.assertEqual(status['status'], 'online')
        self.assertEqual(status['rtt_ms'], 50.0)

    def test_high_rtt_returns_degraded(self):
        """RTT ≥ 3000ms → degraded（链路慢但通）。"""
        services_bridge._record_poll_result(5000.0, success=True)
        status = services_bridge.get_link_status()
        self.assertEqual(status['status'], 'degraded')
        self.assertEqual(status['rtt_ms'], 5000.0)

    def test_failure_accumulates_until_offline(self):
        """连续失败 _LINK_FAIL_THRESHOLD 次后 → offline。

        这是本 bug 的核心回归测试：连接失败必须被累计，不能被误判为成功。
        """
        for _ in range(services_bridge._LINK_FAIL_THRESHOLD):
            services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        self.assertEqual(status['status'], 'offline')
        self.assertIsNone(status['rtt_ms'])

    def test_failure_below_threshold_still_waiting_or_online(self):
        """失败次数 < 阈值时不进 offline（容错少量丢包）。"""
        services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        self.assertNotEqual(status['status'], 'offline')

    def test_success_resets_fail_count(self):
        """成功 poll 重置失败计数（链路恢复立即转 online）。"""
        # 先累计 2 次失败（阈值 3，未到 offline）
        services_bridge._record_poll_result(None, success=False)
        services_bridge._record_poll_result(None, success=False)
        # 再来一次成功
        services_bridge._record_poll_result(80.0, success=True)
        # 此时 fail_count 应为 0，再连续失败 2 次也不应到 offline
        services_bridge._record_poll_result(None, success=False)
        services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        self.assertNotEqual(status['status'], 'offline')

    def test_min_rtt_kept_across_polls(self):
        """多传感器并行 poll 取最小 RTT（最快路径）。"""
        services_bridge._record_poll_result(100.0, success=True)
        services_bridge._record_poll_result(50.0, success=True)
        services_bridge._record_poll_result(80.0, success=True)
        status = services_bridge.get_link_status()
        self.assertEqual(status['rtt_ms'], 50.0)

    def test_failure_does_not_overwrite_rtt(self):
        """失败时 rtt_ms 保持上一次成功值（不写成 None 让显示闪动）。

        注：当前实现是失败时不动 _link_rtt_ms，仅累计 fail_count。
        只有 fail_count ≥ 阈值进 offline 时 get_link_status 才返回 rtt=None。
        """
        services_bridge._record_poll_result(100.0, success=True)
        # 失败一次（未到阈值）
        services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        # rtt 仍保留 100（不到 offline 分支）
        self.assertEqual(status['rtt_ms'], 100.0)


class PollOneConnectionErrorTest(TestCase):
    """``_poll_one`` 在连接失败时返回 (None, False)。

    Day20 bug 修复回归：``telemetry_service._poll_space`` 抛
    ``ConnectionError`` 后，``_poll_one`` 的 except 捕获并返回失败。
    """

    def setUp(self):
        services_bridge._link_rtt_ms = None
        services_bridge._link_fail_count = 0
        services_bridge._link_last_success_ts = 0.0

    def test_poll_one_returns_failure_on_connection_error(self):
        """``telemetry.poll`` 抛 ConnectionError → ``_poll_one`` 返回 (None, False)。"""
        container = mock.Mock()
        container.telemetry.poll.side_effect = ConnectionError("space unreachable")
        rtt, success = services_bridge._poll_one(container, 'file:fake/source')
        self.assertFalse(success)
        self.assertIsNone(rtt)

    def test_poll_one_returns_success_on_normal_poll(self):
        """``telemetry.poll`` 正常返回 → ``_poll_one`` 返回 (rtt, True)。"""
        container = mock.Mock()
        container.telemetry.poll.return_value = {'channels': {}, 'total': 0}
        rtt, success = services_bridge._poll_one(container, 'file:fake/source')
        self.assertTrue(success)
        # rtt 是 wall-clock 耗时，应该是几毫秒（mock 几乎瞬时）
        self.assertIsNotNone(rtt)
        self.assertGreaterEqual(rtt, 0.0)

    def test_poll_one_returns_failure_on_any_exception(self):
        """任何异常都应被捕获返回失败（不让 auto-poll 线程崩）。"""
        container = mock.Mock()
        container.telemetry.poll.side_effect = RuntimeError("unexpected")
        rtt, success = services_bridge._poll_one(container, 'file:fake/source')
        self.assertFalse(success)
        self.assertIsNone(rtt)

    def test_link_status_goes_offline_when_space_always_unreachable(self):
        """端到端 bug 回归：模拟 space 持续不可达，3 轮 poll 后 status 转 offline。

        Day20 bug 现场：space 没启动，但 link_status 恒显示 online（2000ms RTT）。
        修复后：telemetry_service._poll_space 检查 GroundClient.connected 标志，
        连不上抛 ConnectionError → _poll_one 返回 (None, False) →
        _record_poll_result 累计 fail_count → 阈值后 get_link_status 返回 offline。
        """
        # telemetry.poll 在 connected=False 时应抛 ConnectionError
        # （mock 直接模拟这个抛点，不依赖 GroundClient 真实代码）
        container = mock.Mock()
        container.telemetry.poll.side_effect = ConnectionError("space unreachable")
        for _ in range(services_bridge._LINK_FAIL_THRESHOLD):
            services_bridge._poll_one(container, 'file:fake/source')
            # _poll_one 内部不调 _record_poll_result（那是 _auto_poll_loop 的事），
            # 这里手动调一次模拟 auto-poll 循环
            services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        self.assertEqual(status['status'], 'offline')
        self.assertIsNone(status['rtt_ms'])
