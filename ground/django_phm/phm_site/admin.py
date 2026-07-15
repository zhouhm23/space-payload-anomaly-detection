"""SimpleUI admin configuration for PHM data tables.

Two delete actions are provided per table:
  - **「软删除（可恢复）」** — custom action, marks ``is_deleted=1``,
    executes immediately (no confirmation page, because it's reversible).
  - **「删除选中项」** — Django's built-in ``delete_selected`` (with its
    confirmation page), physically removes the rows.  The irreversible
    operation gets the confirmation step.

``get_queryset`` filters out soft-deleted rows so they don't clutter
the admin list view.
"""

from __future__ import annotations

import datetime
import time

from django.contrib import admin

from .models import AlertRecord, DetectionResult, DiagnosisRecord


def _fmt_utc(ts) -> str:
    """Format an epoch-seconds float as a human-readable UTC string.

    Returns '—' for None/empty values. The stored timestamps are unix epoch
    seconds (float); displaying the raw number is unreadable for operators.
    """
    if ts is None or ts == '':
        return '—'
    try:
        return datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError, OSError):
        return str(ts)


class SoftDeleteModelAdmin(admin.ModelAdmin):
    """Base admin: soft-delete action (reversible, no confirm) + built-in
    hard-delete action (irreversible, has Django's confirmation page)."""

    actions = ['soft_delete', 'delete_selected']

    def get_queryset(self, request):
        return super().get_queryset(request).filter(is_deleted=0)

    @admin.action(description='软删除（可恢复）')
    def soft_delete(self, request, queryset):
        """Mark selected rows is_deleted=1.  Reversible via direct SQL,
        so no confirmation page needed."""
        n = queryset.update(is_deleted=1, deleted_at=time.time())
        self.message_user(request, f"已软删除 {n} 条记录（可在数据库中恢复）")


@admin.register(AlertRecord)
class AlertRecordAdmin(SoftDeleteModelAdmin):
    """告警历史表 — 支持筛选 + 直接编辑人工标注."""
    list_display = (
        'id', 'channel', 'alert_type', 'score', 'created_at_display',
        'status', 'llm_verdict', 'human_verdict', 'final_status_display',
    )
    list_display_links = ('id', 'channel')
    list_filter = ('alert_type', 'status', 'llm_verdict', 'human_verdict', 'channel')
    list_editable = ('human_verdict',)
    search_fields = ('channel', 'message')
    readonly_fields = ('final_status', 'raw_snapshot', 'score_snapshot')
    # Note: date_hierarchy omitted — AlertRecord.created_at is a FloatField
    # (unix timestamp), not a DateField/DateTimeField, which would trigger
    # Django admin.E128.
    list_per_page = 50

    @admin.display(description='创建时间(UTC)', ordering='created_at')
    def created_at_display(self, obj):
        return _fmt_utc(obj.created_at)

    @admin.display(description='最终状态', ordering='status')
    def final_status_display(self, obj):
        return obj.final_status


@admin.register(DetectionResult)
class DetectionResultAdmin(SoftDeleteModelAdmin):
    """检测明细表 — 按通道/L1决策筛选."""
    list_display = ('id', 'channel', 'timestamp_display', 'l1_decision', 'final_score', 'ingested_at_display')
    list_display_links = ('id', 'channel')
    list_filter = ('channel', 'l1_decision')
    search_fields = ('channel',)
    list_per_page = 50

    @admin.display(description='时间戳(UTC)', ordering='timestamp')
    def timestamp_display(self, obj):
        return _fmt_utc(obj.timestamp)

    @admin.display(description='入库时间(UTC)', ordering='ingested_at')
    def ingested_at_display(self, obj):
        return _fmt_utc(obj.ingested_at)


@admin.register(DiagnosisRecord)
class DiagnosisRecordAdmin(SoftDeleteModelAdmin):
    """诊断记录表 — 只读浏览（字段只读，但仍可软删除）."""
    list_display = ('id', 'channel', 'alert_type', 'alert_ts_display', 'llm_verdict', 'elapsed_sec', 'created_at_display')
    list_display_links = ('id', 'channel')
    list_filter = ('llm_verdict', 'alert_type', 'channel')
    readonly_fields = (
        'channel', 'alert_type', 'alert_ts', 'diagnosis', 'context_summary',
        'elapsed_sec', 'error', 'llm_verdict', 'created_at',
    )
    list_per_page = 30

    @admin.display(description='告警时间(UTC)', ordering='alert_ts')
    def alert_ts_display(self, obj):
        return _fmt_utc(obj.alert_ts)

    @admin.display(description='创建时间(UTC)', ordering='created_at')
    def created_at_display(self, obj):
        return _fmt_utc(obj.created_at)
