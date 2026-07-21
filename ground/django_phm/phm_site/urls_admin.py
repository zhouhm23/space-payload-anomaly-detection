"""后台 URL 路由（挂载在 /admin/ 下）。

SimpleUI 主入口 + 自定义页面（需求书 §后台）。

自定义页 URL 必须与 settings.SIMPLEUI_CONFIG 中菜单引用的 URL 完全一致：
  /admin/phm_site/{dashboard,alert,recycle,device-tree,settings,models}/

页面按需求书 9 页分批交付，未实现的页暂指向 models_view 占位（避免菜单 404），
待对应页面开发完成后替换为真实 view。
"""
from __future__ import annotations

from django.contrib import admin
from django.urls import path

from . import views_admin

urlpatterns = [
    # 自定义页面必须在 admin.site.urls 之前：admin.site.urls 是一个 URLResolver
    # 会吞掉 /admin/ 下所有未匹配路径返回自己的 404，不会回退到后续 pattern。
    # 自定义页用 /admin/phm_site/<page>/ 前缀，与 admin 内置 URL 不冲突。
    path('phm_site/models/', views_admin.models_view, name='phm_admin_models'),

    # 以下页面尚未实现，暂占位（菜单可点击不 404）
    path('phm_site/dashboard/',   views_admin.models_view, name='phm_admin_dashboard'),
    path('phm_site/alert/',       views_admin.models_view, name='phm_admin_alert'),
    path('phm_site/recycle/',     views_admin.models_view, name='phm_admin_recycle'),
    path('phm_site/device-tree/', views_admin.models_view, name='phm_admin_device_tree'),
    path('phm_site/settings/',    views_admin.models_view, name='phm_admin_settings'),

    # SimpleUI 主体（含登录/首页/用户/审计/数据浏览 ModelAdmin）放最后兜底
    path('', admin.site.urls),
]
