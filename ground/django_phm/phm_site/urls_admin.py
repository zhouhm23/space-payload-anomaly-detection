"""Admin URL routes (mounted under /admin/).

SimpleUI main entry + custom pages (spec admin section — 9 pages total: login / home /
user management / audit log go through SimpleUI defaults; dashboard / alert
management / recycle bin / device tree / system settings / model management
are the custom pages implemented in this file).

Custom-page URLs must match exactly the URLs referenced by the menu in
settings.SIMPLEUI_CONFIG:
  /admin/phm_site/{dashboard,alert,recycle,device-tree,settings,models}/
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

urlpatterns = [
    # Custom pages must precede admin.site.urls: admin.site.urls is a URLResolver
    # that swallows every unmatched path under /admin/ and returns its own 404
    # without falling through to later patterns. Custom pages use the
    # /admin/phm_site/<page>/ prefix, which does not clash with admin's built-in URLs.
    path('phm_site/models/', views_admin.models_view, name='phm_admin_models'),
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

    # SimpleUI main body (login/home/user/audit/data-browsing ModelAdmins) — catch-all, kept last
    path('', admin.site.urls),
]
