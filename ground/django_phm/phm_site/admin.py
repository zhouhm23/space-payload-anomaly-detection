"""SimpleUI admin registrations (minimal scaffold).

Round 1 (1a) of v1.1 only does basic registration + soft-delete filtering.
Later rounds add, per the spec:
- Custom pages (dashboard / device tree / system settings / model management / recycle bin)
- Detail drawer, batch annotation, CSV/JSON export
"""
from __future__ import annotations

from django.contrib import admin
from django.contrib.admin.models import LogEntry

from .models import AlertRecord, DetectionResult, DiagnosisRecord


class SoftDeleteModelAdmin(admin.ModelAdmin):
    """Soft-delete base: ``get_queryset`` filters out deleted rows; ``delete`` becomes a soft delete.

    A later round will add a "purge" action for administrators to physically clean up.
    """
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.filter(is_deleted=0)

    def delete_model(self, request, obj):
        """Single delete → soft delete (UPDATE is_deleted=1)."""
        from phm.database.sqlite_store import SQLiteStore
        # Delegates to SQLiteStore's soft delete (keeps business-table writes single-sourced);
        # round 1 of v1.1 simplifies this to a direct ORM update for now.
        obj.is_deleted = 1
        from time import time
        obj.deleted_at = time()
        obj.save()

    def delete_queryset(self, request, queryset):
        """Batch delete → soft delete."""
        from time import time
        now = time()
        queryset.update(is_deleted=1, deleted_at=now)


@admin.register(DetectionResult)
class DetectionResultAdmin(SoftDeleteModelAdmin):
    list_display = ('channel', 'timestamp', 'l1_decision', 'final_score', 'ingested_at')
    list_filter = ('channel', 'l1_decision')
    search_fields = ('channel',)
    list_per_page = 50
    date_hierarchy = None  # timestamp is a float, not a date


@admin.register(AlertRecord)
class AlertRecordAdmin(SoftDeleteModelAdmin):
    list_display = (
        'id', 'channel', 'alert_type', 'score', 'created_at',
        'status', 'llm_verdict', 'human_verdict',
    )
    list_filter = ('alert_type', 'status', 'llm_verdict', 'human_verdict', 'channel')
    search_fields = ('channel', 'message')
    list_editable = ('human_verdict',)  # edit the human verdict inline on the list page
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


# ── Audit log (spec admin section "Audit log (SimpleUI default)") ─────────────
# Django does not register a ModelAdmin for LogEntry by default (it is not
# visible inside the admin site). The spec calls for "audit log (simpleui
# default)", i.e. make the audit log browsable from the SimpleUI menu, so we
# explicitly register a read-only ModelAdmin (audit logs must not be editable).
@admin.register(LogEntry)
class LogEntryAdmin(admin.ModelAdmin):
    """Read-only audit-log view (records add/change/delete of ModelAdmins inside the admin site).

    Scope note (a boundary the user has accepted): it only records add/change/delete
    of User/Group/business models inside the admin site; custom-page AJAX writes,
    CLI and API calls are not included.
    """
    list_display = (
        'action_time', 'user', 'content_type', 'object_repr',
        'action_flag', 'change_message',
    )
    list_filter = ('action_flag', 'content_type', 'user')
    search_fields = ('object_repr', 'change_message', 'user__username')
    list_per_page = 50
    date_hierarchy = 'action_time'
    # Audit log is read-only — no add/change/delete allowed
    def has_add_permission(self, request):
        return False
    def has_change_permission(self, request, obj=None):
        return False
    def has_delete_permission(self, request, obj=None):
        return False
    # Disable the bulk-action bar (it would otherwise show "M of N selected" +
    # a delete dropdown; the audit log allows no bulk actions, so removing it is cleaner)
    actions = None
