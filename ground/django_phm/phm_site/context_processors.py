"""Django context processors for the PHM front-end.

注入服务端配置到每个模板的 context，让前端无 fetch 延迟读取。
- PHM_THEME: dict，模板可点访问（{{ PHM_THEME.colors.blue }}）
- PHM_THEME_JSON: JSON string，供 <script>window.THEME = ...</script>
"""
from __future__ import annotations

import json

from phm.services.theme_service import get_theme


def theme(request):
    """注入前台主题（颜色/阈值/轮询/图表配置）。

    Vue3 前端从 /api/v2/theme/ 读取（动态可刷），但首屏仍用此 context
    同步注入避免白屏闪烁。
    """
    theme_dict = get_theme().as_dict()
    return {
        "PHM_THEME": theme_dict,
        "PHM_THEME_JSON": json.dumps(theme_dict, ensure_ascii=False),
    }
