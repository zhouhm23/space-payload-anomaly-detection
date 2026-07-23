"""Front-end monitor dashboard routes (mounted under /monitor/).

Dev environment: the Vue3 dev server (:5173) serves the app; Django only
redirects to it.
Production environment: Django serves dist/index.html (the Vue3 build output).

The v1.1 first round (1a) ships a placeholder page; later rounds 1b-1e are
taken over by Vue3.
"""
from __future__ import annotations

from django.urls import path

from . import views_monitor

urlpatterns = [
    path('', views_monitor.monitor_view),
]
