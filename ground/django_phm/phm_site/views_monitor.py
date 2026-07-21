"""前台监控大屏视图（/monitor/）。

v1.1 第一轮（1a）的占位页：Vue3 dev 时跳转到 :5173，生产时 serve dist。
"""
from __future__ import annotations

from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.views.decorators.clickjacking import xframe_options_exempt


@xframe_options_exempt  # 允许 iframe 嵌入（monitor_embed 用）
def monitor_view(request):
    """前台监控大屏入口。

    开发环境（DEBUG=True）：跳转到 vite dev server :5173。
    生产环境：渲染 templates/phm_site/monitor.html（含 Vue3 dist 产物）。
    """
    if settings.DEBUG:
        # 开发环境：直接跳到 vite dev server
        return HttpResponseRedirect('http://127.0.0.1:5173/')

    # 生产环境：渲染模板（Vue3 dist 产物已 collectstatic）
    template = loader.get_template('phm_site/monitor.html')
    return HttpResponse(template.render(request=request))
