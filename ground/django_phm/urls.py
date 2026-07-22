"""Root URLConf for django_phm project.

URL 结构：
- /admin/                  SimpleUI 后台（主体 + 自定义页）
- /api/v2/                 DRF 规范接口（新前端用）
- /api/                    旧视图（过渡保留，与 Day18 CLI 兼容）
- /monitor/                前台监控大屏（Vue3 SPA 入口，生产环境 serve dist/index.html）
- /monitor-dev/            开发环境跳转到 vite dev server :5173（仅 DEBUG=True）
"""
from __future__ import annotations

from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', include('phm_site.urls_admin')),  # 自定义后台 URL（含 admin.site.urls）
    path('api/v2/', include('phm_site.urls_api')),   # DRF 新接口
    path('monitor/', include('phm_site.urls_monitor')),  # 前台大屏
]

# 开发环境：方便跳转到 vite dev server
if settings.DEBUG:
    urlpatterns += [
        path('monitor-dev/', RedirectView.as_view(url='http://127.0.0.1:5173/', permanent=False)),
    ]
