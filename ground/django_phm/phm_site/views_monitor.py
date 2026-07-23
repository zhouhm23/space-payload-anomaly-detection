"""Front-end monitor view (/monitor/).

Round 1 (1a) of v1.1 placeholder page: jumps to :5173 during Vue3 dev, serves
dist in production.
"""
from __future__ import annotations

from django.conf import settings
from django.http import HttpResponse
from django.templatetags.static import static
from django.template import loader
from django.views.decorators.clickjacking import xframe_options_exempt


def _scan_dist_assets():
    """Scan the Vue3 build output dist/assets/ and return (js_url, css_url).

    Vite build output filenames are hashed (index-AbCd1234.js) and must be
    discovered dynamically. Returns (None, None) when not found, in which case
    the template shows a build hint.

    Depends on settings.FRONTEND_DIST (configured in settings.py: when dist
    exists it is added to STATICFILES_DIRS and Django serves that dir).
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
    # Use static() to build the /static/phm_site/dist/assets/xxx URL
    js_url = static(f'phm_site/dist/assets/{js_file}') if js_file else None
    css_url = static(f'phm_site/dist/assets/{css_file}') if css_file else None
    return js_url, css_url


@xframe_options_exempt  # allow iframe embedding (for monitor_embed)
def monitor_view(request):
    """Front-end monitor entry point.

    Unified behaviour (no longer relies on DEBUG to redirect): scans the Vue3
    dist output and serves it if present, otherwise shows a build hint. For
    Vite HMR during development, visit :5173 directly.

    History: the original logic 302'd to :5173 when DEBUG=True, but start_admin
    mode does not start the Vue3 dev server (5173 is unreachable), which left
    /monitor/ on a blank screen.
    """
    js_url, css_url = _scan_dist_assets()
    template = loader.get_template('phm_site/monitor.html')
    return HttpResponse(template.render({
        'dist_js_url': js_url,
        'dist_css_url': css_url,
    }, request=request))
