"""Admin URL routes (mounted under /admin/).

SimpleUI main entry + custom pages (spec admin section — 9 pages total: login / home /
user management / audit log go through SimpleUI defaults; dashboard / alert
management / recycle bin / device tree / system settings / algorithm library
are the custom pages implemented in this file).

Custom-page URLs must match exactly the URLs referenced by the menu in
settings.SIMPLEUI_CONFIG:
  /admin/phm_site/{dashboard,alert,recycle,device-tree,settings,library}/
"""
from __future__ import annotations

from django.contrib import admin
from django.urls import path

from . import views_admin

# ── Admin brand name (replaces Django's default "Django administration") ────
# site_header: top-left brand text + login page title
# site_title : browser tab suffix ("page name | 天地PHM")
# index_title: admin home page heading
# Placed before urlpatterns: the admin.site singleton is already created and
# ready by this point in the Django boot flow.
admin.site.site_header = '天地PHM 管理后台'
admin.site.site_title = '天地PHM'
admin.site.index_title = '天地PHM 运营管理'


def _models_redirect(request):
    """301 permanent redirect: /admin/phm_site/models/ → /admin/phm_site/library/.

    The single-page ``models_view`` was replaced by the 5-sub-menu
    ``library_view`` (v1.2).  Old bookmarks / SimpleUI caches still point at
    ``/models/`` — return a 301 so they update to the new URL and do not
    break.  Permanent (not 302) so search engines and browser caches drop
    the old URL.
    """
    from django.http import HttpResponsePermanentRedirect
    return HttpResponsePermanentRedirect('/admin/phm_site/library/')


urlpatterns = [
    # Custom pages must precede admin.site.urls: admin.site.urls is a URLResolver
    # that swallows every unmatched path under /admin/ and returns its own 404
    # without falling through to later patterns. Custom pages use the
    # /admin/phm_site/<page>/ prefix, which does not clash with admin's built-in URLs.

    # Page 1: Algorithm & model library (5 sub-menus + ground/space dual panel).
    # Replaces the old single-page models_view (kept as a 301 redirect below).
    path('phm_site/library/', views_admin.library_view, name='phm_admin_library'),
    path('phm_site/library/<str:category>/', views_admin.library_view,
         name='phm_admin_library_cat'),
    # Legacy /models/ route → 301 → /library/ (bookmark compat).
    path('phm_site/models/', _models_redirect, name='phm_admin_models'),

    path('phm_site/dashboard/', views_admin.dashboard_view, name='phm_admin_dashboard'),

    # Page 3: recycle bin (superuser-only writes). AJAX endpoints use api/<action>/ sub-paths.
    path('phm_site/recycle/',                views_admin.recycle_view,         name='phm_admin_recycle'),
    path('phm_site/recycle/api/restore/',    views_admin.recycle_restore_api,  name='phm_admin_recycle_restore'),
    path('phm_site/recycle/api/purge/',      views_admin.recycle_purge_api,    name='phm_admin_recycle_purge'),

    # Page 9: permissions explainer (user management + audit log use SimpleUI defaults; only this page is a new static page)
    path('phm_site/permissions/',            views_admin.permissions_view,     name='phm_admin_permissions'),

    # Page 5: system settings (system/theme editable, calibration read-only)
    path('phm_site/settings/',           views_admin.settings_view,      name='phm_admin_settings'),
    path('phm_site/settings/api/save/',  views_admin.settings_save_api,  name='phm_admin_settings_save'),

    # Page 4: alert & warning management (measured alerts only; predicted warnings live on the dashboard)
    path('phm_site/alert/',                          views_admin.alert_view,                 name='phm_admin_alert'),
    path('phm_site/alert/api/detail/<int:alert_id>/', views_admin.alert_detail_api,           name='phm_admin_alert_detail'),
    path('phm_site/alert/api/annotate/',             views_admin.alert_annotate_api,         name='phm_admin_alert_annotate'),
    path('phm_site/alert/api/delete/',               views_admin.alert_delete_api,           name='phm_admin_alert_delete'),
    path('phm_site/alert/api/diagnose/',             views_admin.alert_diagnose_api,         name='phm_admin_alert_diagnose'),
    path('phm_site/alert/api/diagnose_status/',      views_admin.alert_diagnose_status_api,  name='phm_admin_alert_diagnose_status'),
    path('phm_site/alert/api/diagnose_one/<int:alert_id>/', views_admin.alert_diagnose_one_api, name='phm_admin_alert_diagnose_one'),
    path('phm_site/alert/api/export/',               views_admin.alert_export_api,           name='phm_admin_alert_export'),
    path('phm_site/alert/api/create/',               views_admin.alert_create_api,           name='phm_admin_alert_create'),

    # Page 6: device-tree management (service layer unchanged, reuses ConfigService.save)
    path('phm_site/device-tree/',                views_admin.device_tree_view,                name='phm_admin_device_tree'),
    path('phm_site/device-tree/api/save/',       views_admin.device_tree_save_api,            name='phm_admin_device_tree_save'),
    path('phm_site/device-tree/api/channels/',   views_admin.device_tree_space_channels_api,  name='phm_admin_device_tree_channels'),
    path('phm_site/device-tree/api/validate-dsl/', views_admin.device_tree_validate_dsl_api,  name='phm_admin_device_tree_validate_dsl'),

    # Page 7: telemetry data management (single-channel, paginated + chart).
    # Data lives in SQLiteStore per-channel telemetry_* tables, not the ORM.
    path('phm_site/telemetry/',                  views_admin.telemetry_view,            name='phm_admin_telemetry'),
    path('phm_site/telemetry/api/create/',       views_admin.telemetry_create_api,      name='phm_admin_telemetry_create'),
    path('phm_site/telemetry/api/delete/',       views_admin.telemetry_delete_api,      name='phm_admin_telemetry_delete'),
    path('phm_site/telemetry/api/export/',       views_admin.telemetry_export_api,      name='phm_admin_telemetry_export'),
    path('phm_site/telemetry/api/channels/',     views_admin.telemetry_channels_api,    name='phm_admin_telemetry_channels'),

    # SimpleUI main body (login/home/user/audit/data-browsing ModelAdmins) — catch-all, kept last
    path('', admin.site.urls),
]
