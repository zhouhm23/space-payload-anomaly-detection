"""DRF 新接口路由（/api/v2/）。

v1.1 第一轮（1b）大屏数据源接口。
后续轮次按需补：dashboard / alerts CRUD / device-tree CRUD / settings CRUD 等。
"""
from __future__ import annotations

from django.urls import path

from . import views_api

urlpatterns = [
    # 系统类（无需 Container）
    path('ping/', views_api.ping_view),
    path('startup-status/', views_api.startup_status_view),
    path('theme/', views_api.theme_view),

    # 大屏数据源
    path('system-info/', views_api.system_info_view),
    path('device-tree/', views_api.device_tree_view),
    path('window/', views_api.window_view),
    path('alert-points/', views_api.alert_points_view),
    path('alerts/', views_api.alerts_view),
    path('warnings/', views_api.warnings_view),
    path('rul/', views_api.rul_view),
]
