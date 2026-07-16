"""Django context processors for the PHM front-end.

These inject server-side configuration into every template's context so the
front-end can read it synchronously (no fetch delay on page load). The
template renders the JSON into ``window.THEME`` before monitor.js runs.

Registered in settings.py TEMPLATES → OPTIONS → context_processors.
"""

from __future__ import annotations

import json

from phm.services.theme_service import get_theme


def theme(request):
    """Inject the UI theme for both JS and CSS/template consumption.

    Two context variables are exposed:
      * ``PHM_THEME``: the theme as a Python dict — templates can dot-access
        individual values (e.g. ``{{ PHM_THEME.colors.blue }}``) to drive
        inline CSS / dynamic :root variables.
      * ``PHM_THEME_JSON``: the same dict serialised to a JSON string, ready
        for ``<script>window.THEME = {{ PHM_THEME_JSON|safe }};</script>``.

    monitor.js reads ``window.THEME`` with built-in fallbacks, so a missing
    or empty payload never breaks the page.
    """
    theme_dict = get_theme().as_dict()
    return {
        "PHM_THEME": theme_dict,
        "PHM_THEME_JSON": json.dumps(theme_dict, ensure_ascii=False),
    }
