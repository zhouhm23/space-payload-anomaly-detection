"""DRF v2 routes (/api/v2/).

Round 1 (1b) of v1.1: monitor data-source endpoints.
Later rounds add dashboard / alerts CRUD / device-tree CRUD / settings CRUD etc.
"""
from __future__ import annotations

from django.urls import path

from . import views_api

urlpatterns = [
    # System endpoints (no Container needed)
    path('ping/', views_api.ping_view),
    path('startup-status/', views_api.startup_status_view),
    path('theme/', views_api.theme_view),

    # Monitor data sources
    path('system-info/', views_api.system_info_view),
    path('device-tree/', views_api.device_tree_view),
    path('window/', views_api.window_view),
    path('alert-points/', views_api.alert_points_view),
    path('alerts/', views_api.alerts_view),
    path('warnings/', views_api.warnings_view),
    path('rul/', views_api.rul_view),
]
