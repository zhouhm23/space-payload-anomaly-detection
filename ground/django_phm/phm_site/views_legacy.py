"""旧视图过渡保留（/api/）。

v1.1 第一轮（1a）只有 ping 探针，后续轮次按需迁移或回收。
"""
from __future__ import annotations

from django.http import JsonResponse


def ping_view(request):
    return JsonResponse({'status': 'ok', 'legacy': True})
