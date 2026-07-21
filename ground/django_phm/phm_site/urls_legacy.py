"""旧 API 路由过渡保留（/api/）。

v1.1 第一轮（1a）暂留最小代理，让 Day18 CLI（manage.py xxx）和
旧测试（src/ground/django_phm/phm_site/tests/test_views.py）兼容。
后续轮次把 23 个端点迁完 DRF 后，可整体回收此模块。
"""
from __future__ import annotations

from django.urls import path

from . import views_legacy

urlpatterns = [
    path('ping/', views_legacy.ping_view),
]
