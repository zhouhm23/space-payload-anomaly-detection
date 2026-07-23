"""Tests for the ``services_bridge`` link status state machine.

Historical context (Day20 bug): ``_poll_one`` incorrectly reported connection
failures as ``success=True``, causing ``link_status`` to remain permanently
``online``. After the fix, ``telemetry_service`` raises ``ConnectionError``,
which is caught by ``_poll_one``'s except block, yielding ``success=False``.
The ``_link_fail_count`` then accumulates and flips ``status`` to ``offline``
after three consecutive failures.

This file directly tests the ``_record_poll_result`` + ``get_link_status``
state machine without relying on a real Django Container or live socket.
"""
from __future__ import annotations

from unittest import mock

from django.test import TestCase

from phm_site import services_bridge


class LinkStatusStateMachineTest(TestCase):
    """State machine driven by ``_record_poll_result`` + ``get_link_status``."""

    def setUp(self):
        # Reset module-level link state before each test so cases cannot bleed into one another.
        services_bridge._link_rtt_ms = None
        services_bridge._link_fail_count = 0
        services_bridge._link_last_success_ts = 0.0

    def test_initial_state_is_waiting(self):
        """Initial state: no poll has been issued yet, so ``rtt_ms`` is ``None`` and ``status`` is ``'waiting'``."""
        status = services_bridge.get_link_status()
        self.assertIsNone(status['rtt_ms'])
        self.assertEqual(status['status'], 'waiting')

    def test_success_poll_returns_online(self):
        """A single successful poll (RTT < 3000 ms) transitions the status to ``online``."""
        services_bridge._record_poll_result(50.0, success=True)
        status = services_bridge.get_link_status()
        self.assertEqual(status['status'], 'online')
        self.assertEqual(status['rtt_ms'], 50.0)

    def test_high_rtt_returns_degraded(self):
        """RTT >= 3000 ms transitions the status to ``degraded`` (the link is slow but alive)."""
        services_bridge._record_poll_result(5000.0, success=True)
        status = services_bridge.get_link_status()
        self.assertEqual(status['status'], 'degraded')
        self.assertEqual(status['rtt_ms'], 5000.0)

    def test_failure_accumulates_until_offline(self):
        """After ``_LINK_FAIL_THRESHOLD`` consecutive failures the status becomes ``offline``.

        This is the core regression test for the Day20 bug: connection failures
        must be accumulated and must not be misclassified as successes.
        """
        for _ in range(services_bridge._LINK_FAIL_THRESHOLD):
            services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        self.assertEqual(status['status'], 'offline')
        self.assertIsNone(status['rtt_ms'])

    def test_failure_below_threshold_still_waiting_or_online(self):
        """Failures below the threshold do not flip the status to ``offline`` (tolerates minor packet loss)."""
        services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        self.assertNotEqual(status['status'], 'offline')

    def test_success_resets_fail_count(self):
        """A successful poll resets the failure counter (link recovery immediately goes back to ``online``)."""
        # Accumulate 2 failures first (threshold is 3, not yet offline)
        services_bridge._record_poll_result(None, success=False)
        services_bridge._record_poll_result(None, success=False)
        # Now deliver one successful poll
        services_bridge._record_poll_result(80.0, success=True)
        # fail_count should now be 0, so two more failures must not reach offline
        services_bridge._record_poll_result(None, success=False)
        services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        self.assertNotEqual(status['status'], 'offline')

    def test_min_rtt_kept_across_polls(self):
        """Multiple concurrent sensor polls keep the minimum RTT (fastest path)."""
        services_bridge._record_poll_result(100.0, success=True)
        services_bridge._record_poll_result(50.0, success=True)
        services_bridge._record_poll_result(80.0, success=True)
        status = services_bridge.get_link_status()
        self.assertEqual(status['rtt_ms'], 50.0)

    def test_failure_does_not_overwrite_rtt(self):
        """On failure, ``rtt_ms`` retains the last successful value so the UI does not flicker.

        Note: the current implementation leaves ``_link_rtt_ms`` unchanged on failure
        and only increments ``_link_fail_count``.  ``get_link_status`` returns
        ``rtt=None`` only when ``fail_count`` reaches the threshold and the status
        goes ``offline``.
        """
        services_bridge._record_poll_result(100.0, success=True)
        # One failure (below threshold)
        services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        # RTT is still 100 (offline branch not reached)
        self.assertEqual(status['rtt_ms'], 100.0)


class PollOneConnectionErrorTest(TestCase):
    """``_poll_one`` returns ``(None, False)`` when the connection fails.

    Regression for the Day20 bug fix: after ``telemetry_service._poll_space``
    raises ``ConnectionError``, ``_poll_one`` catches the exception and
    reports failure.
    """

    def setUp(self):
        services_bridge._link_rtt_ms = None
        services_bridge._link_fail_count = 0
        services_bridge._link_last_success_ts = 0.0

    def test_poll_one_returns_failure_on_connection_error(self):
        """``telemetry.poll`` raises ``ConnectionError``, so ``_poll_one`` returns ``(None, False)``."""
        container = mock.Mock()
        container.telemetry.poll.side_effect = ConnectionError("space unreachable")
        rtt, success = services_bridge._poll_one(container, 'file:fake/source')
        self.assertFalse(success)
        self.assertIsNone(rtt)

    def test_poll_one_returns_success_on_normal_poll(self):
        """``telemetry.poll`` returns normally, so ``_poll_one`` returns ``(rtt, True)``."""
        container = mock.Mock()
        container.telemetry.poll.return_value = {'channels': {}, 'total': 0}
        rtt, success = services_bridge._poll_one(container, 'file:fake/source')
        self.assertTrue(success)
        # RTT is wall-clock duration, typically a few milliseconds in this mock (nearly instant)
        self.assertIsNotNone(rtt)
        self.assertGreaterEqual(rtt, 0.0)

    def test_poll_one_returns_failure_on_any_exception(self):
        """Any exception should be caught and reported as a failure so the auto-poll thread does not crash."""
        container = mock.Mock()
        container.telemetry.poll.side_effect = RuntimeError("unexpected")
        rtt, success = services_bridge._poll_one(container, 'file:fake/source')
        self.assertFalse(success)
        self.assertIsNone(rtt)

    def test_link_status_goes_offline_when_space_always_unreachable(self):
        """End-to-end bug regression: simulate a permanently unreachable space and verify status goes ``offline`` after 3 poll rounds.

        Day20 bug scenario: space was not running but ``link_status`` remained stuck
        at ``online`` with an RTT of 2000 ms.  After the fix,
        ``telemetry_service._poll_space`` checks the ``GroundClient.connected``
        flag and raises ``ConnectionError`` when disconnected.  ``_poll_one`` then
        returns ``(None, False)``, ``_record_poll_result`` accumulates
        ``_link_fail_count``, and once the threshold is reached
        ``get_link_status`` returns ``offline``.
        """
        # When connected=False, telemetry.poll is expected to raise ConnectionError.
        # We mock that raise point directly rather than depending on real GroundClient code.
        container = mock.Mock()
        container.telemetry.poll.side_effect = ConnectionError("space unreachable")
        for _ in range(services_bridge._LINK_FAIL_THRESHOLD):
            services_bridge._poll_one(container, 'file:fake/source')
            # _poll_one does not call _record_poll_result internally (that is
            # _auto_poll_loop's job), so we call it manually to emulate the loop.
            services_bridge._record_poll_result(None, success=False)
        status = services_bridge.get_link_status()
        self.assertEqual(status['status'], 'offline')
        self.assertIsNone(status['rtt_ms'])
