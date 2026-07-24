"""Page 1b: algorithm & model library page tests.

Coverage:
  (a) 5 sub-menu tabs return the correct card counts (L1=5, L2=1, L3=8,
      forecast=1, special=1 — 16 total).
  (b) ``scan_module_usage`` reads ChannelCalibration (DSL-populated fields)
      and applies default-flow backfill for unconfigured channels.
  (c) Backward-compat: legacy ``@tspulse`` substring in device-tree
      description still resolves when no ChannelCalibration record exists.
  (d) Dual-panel rendering: ground panel full cards, space panel placeholder.
  (e) Read-only: no edit affordances in the rendered HTML.
  (f) Legacy ``/admin/phm_site/models/`` route returns 301 → ``/library/``.
  (g) Showcase ↔ MODEL/FILTER_REGISTRY consistency check returns no warnings.

Tests use Django TestCase.  The Container is mocked where needed; the page
degrades gracefully when the Container is not ready (no 500).
"""
from __future__ import annotations

from unittest import mock

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from phm.algorithm.calibration_config import CalibrationConfig, ChannelCalibration
from phm.algorithm.showcase import (
    SHOWCASE_REGISTRY,
    LAYER_TO_CATEGORY,
    validate_showcase_consistency,
)
from phm_site.views_admin import (
    library_view,
    scan_module_usage,
    _pull_registry_data,
)


# ── Helpers ────────────────────────────────────────────────────────────────
def _make_mock_container(tree=None):
    """Mock Container with a configurable device-tree (for usage-scan fallback)."""
    c = mock.Mock()
    c.config.load.return_value = {
        'device_tree': tree if tree is not None else [],
        'aggregation_strategy': 'min',
    }
    return c


def _patch_container(c):
    return (
        mock.patch('phm_site.services_bridge.get_container', return_value=c),
        mock.patch('phm_site.services_bridge.get_state', return_value='ready'),
    )


def _fake_calibration(channels: dict[str, ChannelCalibration]) -> mock.Mock:
    """Build a fake CalibrationConfig whose .channels / .get mimic a real one.

    ``channels`` maps channel name → ChannelCalibration record.  The fake
    exposes the same surface ``scan_module_usage`` uses (``.channels`` +
    ``.get(name)``).
    """
    cal = mock.Mock()
    cal.channels = list(channels.keys())
    cal.get.side_effect = lambda ch: channels.get(ch)
    return cal


# ── Tests ──────────────────────────────────────────────────────────────────
class LibraryTabCardCountTest(TestCase):
    """Each sub-menu tab renders the right number of cards."""

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_l1_tab_has_five_cards(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertEqual(resp.status_code, 200)
        cards = resp.context['cards']
        self.assertEqual(len(cards), 5)
        # All L1 entries present
        keys = {c['entry'].key for c in cards}
        self.assertEqual(keys, {
            'l1_constant', 'l1_sigma', 'l1_iqr', 'l1_rate', 'l1_setpoint'
        })

    def test_l2_tab_has_one_card(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library_cat', args=['l2']))
        self.assertEqual(resp.status_code, 200)
        cards = resp.context['cards']
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]['entry'].key, 'tspulse')

    def test_l3_tab_has_eight_cards(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library_cat', args=['l3']))
        self.assertEqual(resp.status_code, 200)
        cards = resp.context['cards']
        self.assertEqual(len(cards), 8)
        # L3.5 modules carry is_l35=True
        l35_keys = {c['entry'].key for c in cards if c['is_l35']}
        self.assertEqual(l35_keys, {
            'l3_knee_threshold', 'l3_ema_smoothing', 'l3_persistence'
        })

    def test_forecast_tab_has_one_card(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library_cat', args=['forecast']))
        self.assertEqual(resp.status_code, 200)
        cards = resp.context['cards']
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]['entry'].key, 'ttm_r3')

    def test_special_tab_has_one_card(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library_cat', args=['special']))
        self.assertEqual(resp.status_code, 200)
        cards = resp.context['cards']
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]['entry'].key, 'rul')

    def test_total_showcase_entries_is_sixteen(self):
        """The SHOWCASE_REGISTRY has exactly 16 entries (5+1+8+1+1)."""
        self.assertEqual(len(SHOWCASE_REGISTRY), 16)

    def test_invalid_category_falls_back_to_l1(self):
        self.client.force_login(self.staff)
        resp = self.client.get('/admin/phm_site/library/', {'cat': 'nonexistent'})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['category'], 'l1')

    def test_cat_query_param_switches_tabs(self):
        """The ?cat= query param activates the corresponding tab."""
        self.client.force_login(self.staff)
        resp = self.client.get('/admin/phm_site/library/', {'cat': 'l3'})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['category'], 'l3')
        # The l3 tab carries active=True; others active=False.
        active_tabs = [t for t in resp.context['tabs'] if t['active']]
        self.assertEqual(len(active_tabs), 1)
        self.assertEqual(active_tabs[0]['key'], 'l3')


class ScanModuleUsageTest(TestCase):
    """scan_module_usage reads ChannelCalibration + default-flow backfill."""

    def test_explicit_l1_module_detected(self):
        """A channel with l1_modules=['l1_sigma'] is counted for l1_sigma."""
        cal = _fake_calibration({
            'T-1': ChannelCalibration(l1_modules=['l1_sigma']),
        })
        usage = scan_module_usage(calibration=cal)
        self.assertIn('T-1', usage['l1_sigma'])

    def test_explicit_detector_model_detected(self):
        """A channel with detector_model='tspulse' is counted for tspulse."""
        cal = _fake_calibration({
            'T-1': ChannelCalibration(detector_model='tspulse'),
        })
        usage = scan_module_usage(calibration=cal)
        self.assertIn('T-1', usage['tspulse'])

    def test_default_flow_backfill_l1(self):
        """A channel with no l1_modules is counted for all DEFAULT_L1_MODULES."""
        cal = _fake_calibration({
            'T-1': ChannelCalibration(),  # no overrides
        })
        usage = scan_module_usage(calibration=cal)
        # Defaults: l1_constant, l1_sigma, l1_iqr, l1_rate
        from phm.algorithm.rules import DEFAULT_L1_MODULES
        for m in DEFAULT_L1_MODULES:
            self.assertIn('T-1', usage[m], f"{m} should include T-1 by default")

    def test_default_flow_backfill_detector(self):
        """A channel with no detector_model and skip_detector=False → tspulse."""
        cal = _fake_calibration({
            'T-1': ChannelCalibration(),  # defaults: detector_model=None, skip_detector=False
        })
        usage = scan_module_usage(calibration=cal)
        self.assertIn('T-1', usage['tspulse'])

    def test_skip_detector_suppresses_default_detector(self):
        """skip_detector=True should NOT trigger the default tspulse backfill."""
        cal = _fake_calibration({
            'T-1': ChannelCalibration(skip_detector=True),
        })
        usage = scan_module_usage(calibration=cal)
        self.assertNotIn('T-1', usage['tspulse'])

    def test_default_flow_backfill_l3(self):
        """A channel with no l3_modules is counted for DEFAULT_L3_MODULES."""
        cal = _fake_calibration({
            'T-1': ChannelCalibration(),
        })
        usage = scan_module_usage(calibration=cal)
        from phm.algorithm.rules import DEFAULT_L3_MODULES
        for m in DEFAULT_L3_MODULES:
            self.assertIn('T-1', usage[m])

    def test_explicit_override_skips_default(self):
        """If l1_modules is set, defaults are NOT applied for that layer."""
        cal = _fake_calibration({
            'T-1': ChannelCalibration(l1_modules=['l1_setpoint']),
        })
        usage = scan_module_usage(calibration=cal)
        # l1_setpoint should be present (explicit)
        self.assertIn('T-1', usage['l1_setpoint'])
        # Other default L1 modules should NOT include T-1
        self.assertNotIn('T-1', usage['l1_constant'])
        self.assertNotIn('T-1', usage['l1_sigma'])

    def test_dedup_channel_per_module(self):
        """A channel appears at most once per module even if double-counted."""
        cal = _fake_calibration({
            'T-1': ChannelCalibration(l1_modules=['l1_sigma']),
        })
        usage = scan_module_usage(calibration=cal)
        self.assertEqual(usage['l1_sigma'].count('T-1'), 1)

    def test_empty_calibration_returns_all_keys_empty(self):
        """No channels → every showcase key still present, with empty lists."""
        cal = _fake_calibration({})
        usage = scan_module_usage(calibration=cal)
        for entry in SHOWCASE_REGISTRY:
            self.assertIn(entry.key, usage)
            self.assertEqual(usage[entry.key], [])

    def test_backward_compat_substring_fallback(self):
        """A device-tree with @tspulse but no calibration record still counts."""
        cal = _fake_calibration({})  # empty calibration
        tree = [
            {'type': 'sensor', 'name': 'S1', 'description': '载荷电流 @tspulse 监测'},
        ]
        usage = scan_module_usage(device_tree=tree, calibration=cal)
        self.assertIn('S1', usage['tspulse'])


class LibraryDualPanelTest(TestCase):
    """Ground panel renders cards; space panel is a placeholder."""

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_ground_panel_has_cards(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertEqual(resp.context['ground_cards'], resp.context['cards'])
        self.assertGreater(len(resp.context['ground_cards']), 0)

    def test_space_panel_is_placeholder(self):
        """space_cards is None — the template renders a placeholder block."""
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertIsNone(resp.context['space_cards'])

    def test_space_placeholder_text_rendered(self):
        """The rendered HTML contains the space-segment placeholder copy."""
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertContains(resp, '天基段算法库')
        # Product-facing copy (no internal jargon like "Phase 3" / "src/space").
        self.assertContains(resp, '天地级联')

    def test_ground_panel_header_rendered(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertContains(resp, '地基段算法库')


class LibraryReadOnlyTest(TestCase):
    """The page is read-only: no edit affordances in the rendered HTML."""

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_no_form_or_input_elements(self):
        """No <form>, <input>, or <button> elements for editing."""
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        html = resp.content.decode('utf-8')
        # The page should not contain any submit buttons or forms
        # (the tab links are <a> tags, not form submits).
        self.assertNotIn('<form', html)
        self.assertNotIn('<button', html)
        self.assertNotIn('type="submit"', html)

    def test_readonly_footer_text_rendered(self):
        """Each card carries the read-only footer copy."""
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertContains(resp, '只读')

    def test_readonly_notice_in_header(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertContains(resp, '只读说明')


class LegacyModelsRedirectTest(TestCase):
    """/admin/phm_site/models/ → 301 → /admin/phm_site/library/."""

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_models_route_returns_301(self):
        """The old /models/ URL returns a 301 permanent redirect."""
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_models'))
        self.assertEqual(resp.status_code, 301)

    def test_models_redirect_target_is_library(self):
        """The redirect Location points at /admin/phm_site/library/."""
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_models'))
        self.assertIn('/admin/phm_site/library/', resp['Location'])


class ShowcaseConsistencyTest(TestCase):
    """Every ShowcaseEntry key resolves in MODEL_REGISTRY or FILTER_REGISTRY."""

    def test_no_warnings_when_registries_in_sync(self):
        """validate_showcase_consistency returns no warnings against the
        shipped MODEL_REGISTRY + FILTER_REGISTRY."""
        warnings = validate_showcase_consistency()
        self.assertEqual(warnings, [])

    def test_every_entry_key_resolves(self):
        """Every showcase key has a matching MODEL or FILTER registry entry."""
        from phm.algorithm._registry import MODEL_REGISTRY
        from phm.algorithm.rules import FILTER_REGISTRY
        for entry in SHOWCASE_REGISTRY:
            if entry.is_model:
                self.assertIn(entry.key, MODEL_REGISTRY,
                              f"model entry {entry.key!r} not in MODEL_REGISTRY")
            else:
                self.assertIn(entry.key, FILTER_REGISTRY,
                              f"filter entry {entry.key!r} not in FILTER_REGISTRY")

    def test_layer_to_category_covers_all_layers(self):
        """Every showcase entry's layer maps to a sub-menu category."""
        for entry in SHOWCASE_REGISTRY:
            self.assertIn(entry.layer, LAYER_TO_CATEGORY,
                          f"layer {entry.layer!r} has no category mapping")


class PullRegistryDataTest(TestCase):
    """_pull_registry_data pulls the right fields for model vs algorithm cards."""

    def test_model_entry_has_hub_id(self):
        """A model entry pulls hub_id + context_length from MODEL_REGISTRY."""
        from phm.algorithm.showcase import ShowcaseEntry
        entry = next(e for e in SHOWCASE_REGISTRY if e.key == 'tspulse')
        data = _pull_registry_data(entry)
        self.assertIn('hub_id', data)
        self.assertIn('context_length', data)
        self.assertEqual(data['context_length'], 512)

    def test_algorithm_entry_has_rule_class(self):
        """An algorithm entry pulls the rule class name from FILTER_REGISTRY."""
        from phm.algorithm.showcase import ShowcaseEntry
        entry = next(e for e in SHOWCASE_REGISTRY if e.key == 'l1_sigma')
        data = _pull_registry_data(entry)
        self.assertEqual(data['rule_class'], 'L1SigmaRule')

    def test_rul_entry_has_no_hub_id(self):
        """RUL has empty hub_id (local weights) — key absent, not None."""
        entry = next(e for e in SHOWCASE_REGISTRY if e.key == 'rul')
        data = _pull_registry_data(entry)
        self.assertNotIn('hub_id', data)
        self.assertIn('context_length', data)


class LibraryAccessControlTest(TestCase):
    """Page access control (login gate)."""

    def setUp(self):
        self.client = Client()

    def test_anonymous_redirects_to_login(self):
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/admin/login/', resp['Location'])

    def test_staff_can_access(self):
        staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.client.force_login(staff)
        resp = self.client.get(reverse('phm_admin_library'))
        self.assertEqual(resp.status_code, 200)


class LibraryContainerNotReadyTest(TestCase):
    """The page still renders when the Container is not ready."""

    def test_initializing_state_renders_cards(self):
        """Even with no Container, the page renders cards (no 500)."""
        staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )
        self.client.force_login(staff)
        with mock.patch('phm_site.views_admin.services_bridge') as mock_sb:
            mock_sb.get_state.return_value = 'initializing'
            mock_sb.get_init_error.return_value = None
            resp = self.client.get(reverse('phm_admin_library'))
        self.assertEqual(resp.status_code, 200)
        # Cards still rendered (showcase metadata is static).
        self.assertGreater(len(resp.context['cards']), 0)


class LibraryL35TagTest(TestCase):
    """L3.5 cards render with the '后处理增强' tag."""

    def setUp(self):
        self.client = Client()
        self.staff = User.objects.create_user(
            username='staff1', password='pw', is_staff=True
        )

    def test_l35_tag_rendered_on_l3_tab(self):
        """The L3 sub-menu renders the '后处理增强' badge for L3.5 cards."""
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library_cat', args=['l3']))
        self.assertContains(resp, '后处理增强')

    def test_l35_tag_not_rendered_on_l1_tab(self):
        """The L1 sub-menu has no L3.5 cards → no '后处理增强' badge on cards.

        The read-only notice paragraph mentions '后处理增强' in prose, so we
        check for the badge-specific markup (phm-badge-yellow + the tag text
        inside a card) instead of bare substring presence.
        """
        self.client.force_login(self.staff)
        resp = self.client.get(reverse('phm_admin_library'))
        html = resp.content.decode('utf-8')
        # The L3.5 badge is rendered as:
        #   <span class="phm-badge phm-badge-yellow" ...>后处理增强</span>
        # On the L1 tab no card carries it, so this exact combo is absent.
        self.assertNotIn('phm-badge-yellow" title="Layer 3.5', html)
        # And none of the L1 cards have is_l35=True.
        for card in resp.context['cards']:
            self.assertFalse(card['is_l35'])
