"""Django ORM models for the 3 fixed tables (alert/detection/diagnosis).

These mirror the schema in phm.database.sqlite_store._SCHEMA so the ORM
and SQLiteStore share the same phm.db file.  ORM is read-only for these
tables; writes still go through SQLiteStore's background flush thread.
"""

from __future__ import annotations

from django.db import models


class DetectionResult(models.Model):
    """Per-block three-layer cascade detection results.

    Maps to the raw-SQL table ``detection_results`` created by
    ``phm.database.sqlite_store.SQLiteStore`` so SimpleUI admin reads the
    SAME table that the telemetry pipeline writes to. ORM is read-mostly
    here; writes still go through SQLiteStore's background flush thread.
    """

    channel = models.CharField('通道', max_length=64)
    timestamp = models.FloatField('时间戳')
    l1_decision = models.CharField('L1决策', max_length=16, null=True, blank=True)
    l1_score = models.FloatField('L1分数', null=True, blank=True)
    l1_detail = models.TextField('L1明细', null=True, blank=True)
    l2_score = models.FloatField('L2分数', null=True, blank=True)
    l3_score = models.FloatField('L3分数', null=True, blank=True)
    l3_rules = models.TextField('L3规则', null=True, blank=True)
    final_score = models.FloatField('最终分数', null=True, blank=True)
    ingested_at = models.FloatField('入库时间')
    is_deleted = models.IntegerField('已删除', default=0)
    deleted_at = models.FloatField('删除时间', null=True, blank=True)

    class Meta:
        verbose_name = '检测明细'
        verbose_name_plural = '检测明细'
        db_table = 'detection_results'
        # Index name must match SQLiteStore._SCHEMA to avoid Django trying
        # to drop/recreate it on migrate.
        indexes = [
            models.Index(fields=['channel', 'timestamp'], name='idx_det_channel_time'),
        ]

    def __str__(self):
        return f"Detection({self.channel}, {self.timestamp}, {self.final_score})"


class AlertRecord(models.Model):
    """Measured + predicted alerts with lifecycle + verdict dimensions.

    Maps to the raw-SQL table ``alert_records`` (see DetectionResult doc
    for the shared-table rationale).
    """

    channel = models.CharField('通道', max_length=64)
    alert_type = models.CharField('告警类型', max_length=16)  # 'measured' | 'predicted'
    score = models.FloatField('分数', null=True, blank=True)
    message = models.TextField('消息', null=True, blank=True)
    created_at = models.FloatField('创建时间')
    status = models.CharField('状态', max_length=16, default='active')
    verified_at = models.FloatField('核验时间', null=True, blank=True)
    llm_verdict = models.CharField('LLM裁决', max_length=16, null=True, blank=True)
    human_verdict = models.CharField('人工裁决', max_length=16, null=True, blank=True)
    raw_snapshot = models.TextField('原始波形快照', null=True, blank=True)
    score_snapshot = models.TextField('分数快照', null=True, blank=True)
    ingested_at = models.FloatField('入库时间')
    is_deleted = models.IntegerField('已删除', default=0)
    deleted_at = models.FloatField('删除时间', null=True, blank=True)

    class Meta:
        verbose_name = '告警记录'
        verbose_name_plural = '告警记录'
        db_table = 'alert_records'
        indexes = [
            models.Index(fields=['channel', 'created_at'], name='idx_alert_channel_time'),
        ]

    @property
    def final_status(self) -> str:
        """Derived: human_verdict > llm_verdict > status."""
        if self.human_verdict:
            return self.human_verdict
        if self.llm_verdict:
            return self.llm_verdict
        return self.status

    def __str__(self):
        return f"Alert({self.channel}, {self.alert_type}, {self.final_status})"


class DiagnosisRecord(models.Model):
    """LLM diagnosis cache (one row per unique alert).

    Maps to the raw-SQL table ``diagnosis_records``.
    """

    channel = models.CharField('通道', max_length=64)
    alert_type = models.CharField('告警类型', max_length=16)
    alert_ts = models.FloatField('告警时间戳')
    diagnosis = models.TextField('诊断报告', null=True, blank=True)
    context_summary = models.TextField('上下文摘要', null=True, blank=True)
    elapsed_sec = models.FloatField('耗时(秒)', null=True, blank=True)
    error = models.TextField('错误', null=True, blank=True)
    llm_verdict = models.CharField('LLM裁决', max_length=16, null=True, blank=True)
    created_at = models.FloatField('创建时间')
    is_deleted = models.IntegerField('已删除', default=0)
    deleted_at = models.FloatField('删除时间', null=True, blank=True)

    class Meta:
        verbose_name = '诊断记录'
        verbose_name_plural = '诊断记录'
        db_table = 'diagnosis_records'
        unique_together = ('channel', 'alert_type', 'alert_ts')

    def __str__(self):
        return f"Diagnosis({self.channel}, {self.alert_type}, {self.alert_ts})"
