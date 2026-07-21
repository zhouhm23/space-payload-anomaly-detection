"""前台监控大屏路由（/monitor/）。

开发环境：Vue3 dev server (:5173)，Django 只做跳转。
生产环境：serve dist/index.html（Vue3 build 产物）。

v1.1 第一轮（1a）放占位页，后续 1b-1e 由 Vue3 接管。
"""
from __future__ import annotations

from django.urls import path

from . import views_monitor

urlpatterns = [
    path('', views_monitor.monitor_view),
]
