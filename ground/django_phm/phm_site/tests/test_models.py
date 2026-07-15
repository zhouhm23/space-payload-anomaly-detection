"""Tests for Django ORM models (3 fixed tables)."""
import pytest
from django.test import TestCase

from phm_site.models import AlertRecord, DetectionResult, DiagnosisRecord


@pytest.mark.django_db
class TestAlertRecord(TestCase):
    def test_create_alert(self):
        alert = AlertRecord.objects.create(
            channel="C-1", alert_type="measured", score=0.85,
            message="test", created_at=1700000000.0, ingested_at=1700000001.0,
        )
        assert alert.id is not None
        assert alert.status == "active"
        assert alert.llm_verdict is None
        assert alert.human_verdict is None

    def test_final_status_human_priority(self):
        alert = AlertRecord.objects.create(
            channel="C-1", alert_type="measured", created_at=1700000000.0,
            ingested_at=1700000001.0, status="pending",
            llm_verdict="false_alarm", human_verdict="real",
        )
        assert alert.final_status == "real"

    def test_final_status_llm_fallback(self):
        alert = AlertRecord.objects.create(
            channel="C-1", alert_type="measured", created_at=1700000000.0,
            ingested_at=1700000001.0, status="pending", llm_verdict="uncertain",
        )
        assert alert.final_status == "uncertain"

    def test_final_status_status_fallback(self):
        alert = AlertRecord.objects.create(
            channel="C-1", alert_type="measured", created_at=1700000000.0,
            ingested_at=1700000001.0, status="confirmed",
        )
        assert alert.final_status == "confirmed"


@pytest.mark.django_db
class TestDetectionResult(TestCase):
    def test_create_detection(self):
        det = DetectionResult.objects.create(
            channel="C-1", timestamp=1700000000.0,
            l1_decision="alert", l1_score=0.9, l2_score=0.8, l3_score=0.7,
            final_score=0.85, ingested_at=1700000001.0,
        )
        assert det.id is not None
        assert det.l1_decision == "alert"


@pytest.mark.django_db
class TestDiagnosisRecord(TestCase):
    def test_create_diagnosis(self):
        diag = DiagnosisRecord.objects.create(
            channel="C-1", alert_type="measured", alert_ts=1700000000.0,
            diagnosis="## 报告\n异常", elapsed_sec=2.5, llm_verdict="real",
            created_at=1700000001.0,
        )
        assert diag.id is not None

    def test_unique_constraint(self):
        DiagnosisRecord.objects.create(
            channel="C-1", alert_type="measured", alert_ts=1700000000.0,
            created_at=1700000001.0,
        )
        from django.db import IntegrityError
        with pytest.raises(IntegrityError):
            DiagnosisRecord.objects.create(
                channel="C-1", alert_type="measured", alert_ts=1700000000.0,
                created_at=1700000002.0,
            )


@pytest.mark.django_db
class TestSoftDeleteColumns(TestCase):
    """Verify the is_deleted / deleted_at fields exist with correct defaults."""

    def test_alert_has_soft_delete_fields(self):
        a = AlertRecord.objects.create(
            channel="C-1", alert_type="measured", created_at=1700000000.0,
            ingested_at=1700000001.0,
        )
        assert a.is_deleted == 0
        assert a.deleted_at is None

    def test_detection_has_soft_delete_fields(self):
        d = DetectionResult.objects.create(
            channel="C-1", timestamp=1700000000.0, ingested_at=1700000001.0,
        )
        assert d.is_deleted == 0
        assert d.deleted_at is None

    def test_diagnosis_has_soft_delete_fields(self):
        d = DiagnosisRecord.objects.create(
            channel="C-1", alert_type="measured", alert_ts=1700000000.0,
            created_at=1700000001.0,
        )
        assert d.is_deleted == 0
        assert d.deleted_at is None

    def test_mark_deleted_then_query_excludes(self):
        """ORM queries should be able to filter is_deleted=0."""
        AlertRecord.objects.create(
            channel="C-1", alert_type="measured", created_at=1700000000.0,
            ingested_at=1700000001.0, is_deleted=1,
        )
        AlertRecord.objects.create(
            channel="C-2", alert_type="measured", created_at=1700000000.0,
            ingested_at=1700000001.0,
        )
        visible = AlertRecord.objects.filter(is_deleted=0)
        assert visible.count() == 1
        assert visible.first().channel == "C-2"
        all_rows = AlertRecord.objects.all()
        assert all_rows.count() == 2
