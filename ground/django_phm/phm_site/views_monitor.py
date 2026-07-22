"""前台监控大屏视图（/monitor/）。

v1.1 第一轮（1a）的占位页：Vue3 dev 时跳转到 :5173，生产时 serve dist。
"""
from __future__ import annotations

from django.conf import settings
from django.http import HttpResponse
from django.templatetags.static import static
from django.template import loader
from django.views.decorators.clickjacking import xframe_options_exempt


def _scan_dist_assets():
    """扫描 Vue3 build 产物 dist/assets/，返回 (js_url, css_url)。

    Vite 构建产物文件名带 hash（index-AbCd1234.js），需动态发现。
    找不到时返回 (None, None)，模板据此显示构建提示。

    依赖 settings.FRONTEND_DIST（settings.py 已配置：dist 存在则加入
    STATICFILES_DIRS，Django 会 serve 该目录）。
    """
    dist = getattr(settings, 'FRONTEND_DIST', None)
    if not dist:
        return None, None
    assets_dir = dist / 'assets'
    if not assets_dir.is_dir():
        return None, None
    js_file = None
    css_file = None
    for p in assets_dir.iterdir():
        name = p.name
        if name.startswith('index-') and name.endswith('.js') and js_file is None:
            js_file = name
        elif name.startswith('index-') and name.endswith('.css') and css_file is None:
            css_file = name
    # 用 static() 生成 /static/phm_site/dist/assets/xxx URL
    js_url = static(f'phm_site/dist/assets/{js_file}') if js_file else None
    css_url = static(f'phm_site/dist/assets/{css_file}') if css_file else None
    return js_url, css_url


@xframe_options_exempt  # 允许 iframe 嵌入（monitor_embed 用）
def monitor_view(request):
    """前台监控大屏入口。

    统一行为（不再依赖 DEBUG 跳转）：扫描 Vue3 dist 产物，有则 serve，
    无则显示构建提示。开发时若想用 Vite HMR，手动访问 :5173 即可。

    历史问题：原逻辑 DEBUG=True 时 302 跳 :5173，但 start_admin 模式
    不启动 Vue3 dev server（5173 不可达），导致 /monitor/ 白屏打不开。
    """
    js_url, css_url = _scan_dist_assets()
    template = loader.get_template('phm_site/monitor.html')
    return HttpResponse(template.render({
        'dist_js_url': js_url,
        'dist_css_url': css_url,
    }, request=request))
