"""Django ORM 模型。

设计原则：
- 现有业务表（alert_records/detection_results/diagnosis_records）通过 db_table
  映射到 SQLiteStore 的真实表，ORM 与 SQLiteStore 共用同一份 phm.db。
- 写入仍走 SQLiteStore 后台 flush 线程；ORM 主要用于后台浏览/筛选/CRUD。
- v1.1 新增模型（设备树/系统设置/审计日志）走标准 Django migration。

v1.1 第一轮（1a）只迁移 3 个现有业务表 + AlertRecord 的 verdict 四维度。
后续轮次按需新增设备树/设置/审计模型。
"""
from __future__ import annotations

from django.db import models

# verdict 四维度共用 choices（与 phm.database.warning_store._VALID_VERDICTS 对齐）
VERDICT_CHOICES = [
    ('',            '—（未标注）—'),
    ('real',        '真实异常'),
    ('false_alarm', '误报'),
    ('uncertain',   '不确定'),
]

# 告警核验状态 choices（alert_records.status）
ALERT_STATUS_CHOICES = [
    ('active',      '活跃'),
    ('pending',     '待定'),
    ('confirmed',   '已证实'),
    ('false',       '已证伪'),
    ('unverifiable', '无法核验'),
]


class DetectionResult(models.Model):
    """三层级联检测逐块结果。

    db_table = detection_results（SQLiteStore 真实表名）。ORM 只读为主，
    写入走 SQLiteStore 后台 flush 线程。
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
        verbose_name_plural = verbose_name
        db_table = 'detection_results'
        managed = False  # 表由 SQLiteStore 创建/维护，Django 不接管 schema
        indexes = [
            models.Index(fields=['channel', 'timestamp'], name='idx_det_channel_time'),
        ]

    def __str__(self):
        return f"Detection({self.channel}, {self.timestamp}, {self.final_score})"


class AlertRecord(models.Model):
    """实测 + 预测告警（含四维度 verdict + 快照）。

    db_table = alert_records。alert_type 区分 measured/predicted。
    """
    channel = models.CharField('通道', max_length=64)
    alert_type = models.CharField('告警类型', max_length=16)  # measured | predicted
    score = models.FloatField('分数', null=True, blank=True)
    message = models.TextField('消息', null=True, blank=True)
    created_at = models.FloatField('创建时间')
    status = models.CharField('核验状态', max_length=16, default='active', choices=ALERT_STATUS_CHOICES)
    verified_at = models.FloatField('核验时间', null=True, blank=True)
    llm_verdict = models.CharField('LLM裁决', max_length=16, null=True, blank=True, choices=VERDICT_CHOICES)
    human_verdict = models.CharField('人工裁决', max_length=16, null=True, blank=True, choices=VERDICT_CHOICES)
    raw_snapshot = models.TextField('原始波形快照', null=True, blank=True)
    score_snapshot = models.TextField('分数快照', null=True, blank=True)
    ingested_at = models.FloatField('入库时间')
    is_deleted = models.IntegerField('已删除', default=0)
    deleted_at = models.FloatField('删除时间', null=True, blank=True)

    class Meta:
        verbose_name = '告警记录'
        verbose_name_plural = verbose_name
        db_table = 'alert_records'
        managed = False  # 表由 SQLiteStore 创建/维护，Django 不接管 schema
        indexes = [
            models.Index(fields=['channel', 'created_at'], name='idx_alert_channel_time'),
        ]

    @property
    def final_status(self) -> str:
        """派生综合状态：人工 > LLM > 核验状态。"""
        if self.human_verdict:
            return self.human_verdict
        if self.llm_verdict:
            return self.llm_verdict
        return self.status

    def __str__(self):
        return f"Alert({self.channel}, {self.alert_type}, {self.final_status})"


class DiagnosisRecord(models.Model):
    """LLM 诊断缓存（一条 alert 一行，唯一键 channel+alert_type+alert_ts）。"""
    channel = models.CharField('通道', max_length=64)
    alert_type = models.CharField('告警类型', max_length=16)
    alert_ts = models.FloatField('告警时间戳')
    diagnosis = models.TextField('诊断报告', null=True, blank=True)
    context_summary = models.TextField('上下文摘要', null=True, blank=True)
    elapsed_sec = models.FloatField('耗时(秒)', null=True, blank=True)
    error = models.TextField('错误', null=True, blank=True)
    llm_verdict = models.CharField('LLM裁决', max_length=16, null=True, blank=True, choices=VERDICT_CHOICES)
    created_at = models.FloatField('创建时间')
    is_deleted = models.IntegerField('已删除', default=0)
    deleted_at = models.FloatField('删除时间', null=True, blank=True)

    class Meta:
        verbose_name = '诊断记录'
        verbose_name_plural = verbose_name
        db_table = 'diagnosis_records'
        managed = False  # 表由 SQLiteStore 创建/维护，Django 不接管 schema
        unique_together = ('channel', 'alert_type', 'alert_ts')

    def __str__(self):
        return f"Diagnosis({self.channel}, {self.alert_type}, {self.alert_ts})"
