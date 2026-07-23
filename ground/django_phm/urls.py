"""Root URLConf for django_phm project.

URL structure:
- /admin/                  SimpleUI admin (body + custom pages)
- /api/v2/                 DRF v2 endpoints (used by the new front-end)
- /api/                    legacy views (kept for transition, Day18 CLI compatible)
- /monitor/                front-end monitor (Vue3 SPA entry; production serves dist/index.html)
- /monitor-dev/            dev redirect to the vite dev server :5173 (DEBUG=True only)
"""
from __future__ import annotations

from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', include('phm_site.urls_admin')),  # custom admin URLs (includes admin.site.urls)
    path('api/v2/', include('phm_site.urls_api')),   # DRF v2 endpoints
    path('monitor/', include('phm_site.urls_monitor')),  # front-end monitor
]

# Dev convenience: redirect to the vite dev server
if settings.DEBUG:
    urlpatterns += [
        path('monitor-dev/', RedirectView.as_view(url='http://127.0.0.1:5173/', permanent=False)),
    ]
