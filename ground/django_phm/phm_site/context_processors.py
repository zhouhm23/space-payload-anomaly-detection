"""Django context processors for the PHM front-end.

Injects server-side config into every template's context so the front-end can
read it with no fetch latency.
- PHM_THEME: dict, dotted access in templates ({{ PHM_THEME.colors.blue }})
- PHM_THEME_JSON: JSON string, for <script>window.THEME = ...</script>
"""
from __future__ import annotations

import json

from phm.services.theme_service import get_theme


def theme(request):
    """Inject the front-end theme (colors / thresholds / polling / chart config).

    The Vue3 front-end reads it from /api/v2/theme/ (dynamic, refreshable), but
    the first paint still uses this synchronously-injected context to avoid a
    white-screen flash.
    """
    theme_dict = get_theme().as_dict()
    return {
        "PHM_THEME": theme_dict,
        "PHM_THEME_JSON": json.dumps(theme_dict, ensure_ascii=False),
    }
