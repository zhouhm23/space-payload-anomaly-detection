"""后台 URL 路由（挂载在 /admin/ 下）。

SimpleUI 主入口 + 自定义页面（需求书 §后台 共 9 页：登录 / 首页 / 用户管理 /
审计日志走 SimpleUI 默认；仪表盘 / 告警管理 / 回收站 / 设备树 / 系统设置 /
模型管理为本文件实现的自定义页面）。

自定义页 URL 必须与 settings.SIMPLEUI_CONFIG 中菜单引用的 URL 完全一致：
  /admin/phm_site/{dashboard,alert,recycle,device-tree,settings,models}/
"""
from __future__ import annotations

from django.contrib import admin
from django.urls import path

from . import views_admin

# ── 后台品牌名（替代 Django 默认 "Django administration"）─────────────────
# site_header: 左上角品牌文字 + 登录页标题
# site_title : 浏览器标签页后缀（"页面名 | 天地PHM"）
# index_title: admin 首页大标题
# 放在 urlpatterns 之前：admin.site 单例此时已由 Django 启动流程创建就绪。
admin.site.site_header = '天地PHM 管理后台'
admin.site.site_title = '天地PHM'
admin.site.index_title = '天地PHM 运营管理'

urlpatterns = [
    # 自定义页面必须在 admin.site.urls 之前：admin.site.urls 是一个 URLResolver
    # 会吞掉 /admin/ 下所有未匹配路径返回自己的 404，不会回退到后续 pattern。
    # 自定义页用 /admin/phm_site/<page>/ 前缀，与 admin 内置 URL 不冲突。
    path('phm_site/models/', views_admin.models_view, name='phm_admin_models'),
    path('phm_site/dashboard/', views_admin.dashboard_view, name='phm_admin_dashboard'),

    # 第 3 页：回收站（仅超管可改）。AJAX 端点用 api/<action>/ 子路径。
    path('phm_site/recycle/',                views_admin.recycle_view,         name='phm_admin_recycle'),
    path('phm_site/recycle/api/restore/',    views_admin.recycle_restore_api,  name='phm_admin_recycle_restore'),
    path('phm_site/recycle/api/purge/',      views_admin.recycle_purge_api,    name='phm_admin_recycle_purge'),

    # 第 9 页：权限说明（用户管理+审计日志走 SimpleUI 默认，仅本页为新增静态页）
    path('phm_site/permissions/',            views_admin.permissions_view,     name='phm_admin_permissions'),

    # 第 5 页：系统设置（system/theme 可改，calibration 只读）
    path('phm_site/settings/',           views_admin.settings_view,      name='phm_admin_settings'),
    path('phm_site/settings/api/save/',  views_admin.settings_save_api,  name='phm_admin_settings_save'),

    # 第 4 页：告警和预警管理（仅 measured 告警；predicted 在仪表盘）
    path('phm_site/alert/',                          views_admin.alert_view,                 name='phm_admin_alert'),
    path('phm_site/alert/api/detail/<int:alert_id>/', views_admin.alert_detail_api,           name='phm_admin_alert_detail'),
    path('phm_site/alert/api/annotate/',             views_admin.alert_annotate_api,         name='phm_admin_alert_annotate'),
    path('phm_site/alert/api/delete/',               views_admin.alert_delete_api,           name='phm_admin_alert_delete'),
    path('phm_site/alert/api/diagnose/',             views_admin.alert_diagnose_api,         name='phm_admin_alert_diagnose'),
    path('phm_site/alert/api/diagnose_status/',      views_admin.alert_diagnose_status_api,  name='phm_admin_alert_diagnose_status'),
    path('phm_site/alert/api/export/',               views_admin.alert_export_api,           name='phm_admin_alert_export'),
    path('phm_site/alert/api/create/',               views_admin.alert_create_api,           name='phm_admin_alert_create'),

    # 第 6 页：设备树管理（service 层零改动，复用 ConfigService.save）
    path('phm_site/device-tree/',                views_admin.device_tree_view,                name='phm_admin_device_tree'),
    path('phm_site/device-tree/api/save/',       views_admin.device_tree_save_api,            name='phm_admin_device_tree_save'),
    path('phm_site/device-tree/api/channels/',   views_admin.device_tree_space_channels_api,  name='phm_admin_device_tree_channels'),

    # SimpleUI 主体（含登录/首页/用户/审计/数据浏览 ModelAdmin）放最后兜底
    path('', admin.site.urls),
]
