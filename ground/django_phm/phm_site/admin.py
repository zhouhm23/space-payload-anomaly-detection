"""SimpleUI 后台注册（最小骨架）。

v1.1 第一轮（1a）只做基本注册 + 软删除过滤。后续轮次按需求书补：
- 自定义页面（仪表盘/设备树/系统设置/模型管理/回收站）
- 详情抽屉、批量标注、导出 CSV/JSON
"""
from __future__ import annotations

from django.contrib import admin

from .models import AlertRecord, DetectionResult, DiagnosisRecord


class SoftDeleteModelAdmin(admin.ModelAdmin):
    """软删除基类：get_queryset 过滤已删除，delete 改软删除。

    后续轮次补「彻底删除」action 供管理员物理清理。
    """
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.filter(is_deleted=0)

    def delete_model(self, request, obj):
        """单条删除 → 软删除（UPDATE is_deleted=1）。"""
        from phm.database.sqlite_store import SQLiteStore
        # 委托给 SQLiteStore 的软删除（保持业务表写入同源）
        # v1.1 第一轮先简化为 ORM 直接 update
        obj.is_deleted = 1
        from time import time
        obj.deleted_at = time()
        obj.save()

    def delete_queryset(self, request, queryset):
        """批量删除 → 软删除。"""
        from time import time
        now = time()
        queryset.update(is_deleted=1, deleted_at=now)


@admin.register(DetectionResult)
class DetectionResultAdmin(SoftDeleteModelAdmin):
    list_display = ('channel', 'timestamp', 'l1_decision', 'final_score', 'ingested_at')
    list_filter = ('channel', 'l1_decision')
    search_fields = ('channel',)
    list_per_page = 50
    date_hierarchy = None  # timestamp 是 float 不是 date


@admin.register(AlertRecord)
class AlertRecordAdmin(SoftDeleteModelAdmin):
    list_display = (
        'id', 'channel', 'alert_type', 'score', 'created_at',
        'status', 'llm_verdict', 'human_verdict',
    )
    list_filter = ('alert_type', 'status', 'llm_verdict', 'human_verdict', 'channel')
    search_fields = ('channel', 'message')
    list_editable = ('human_verdict',)  # 列表页直接改人工裁决
    list_per_page = 50
    readonly_fields = ('raw_snapshot', 'score_snapshot')

    @admin.display(description='综合状态')
    def final_status_display(self, obj):
        return obj.final_status


@admin.register(DiagnosisRecord)
class DiagnosisRecordAdmin(SoftDeleteModelAdmin):
    list_display = ('channel', 'alert_type', 'alert_ts', 'llm_verdict', 'elapsed_sec', 'created_at')
    list_filter = ('alert_type', 'llm_verdict')
    search_fields = ('channel', 'diagnosis')
    list_per_page = 50
    readonly_fields = ('diagnosis', 'context_summary')
