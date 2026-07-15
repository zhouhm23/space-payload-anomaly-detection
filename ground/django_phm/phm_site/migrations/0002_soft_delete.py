"""Add is_deleted/deleted_at soft-delete columns to the 3 business tables.

The actual ALTER TABLE is also performed by SQLiteStore._migrate_soft_delete_columns()
at runtime for backward compat.  This migration declares the model state change
and is a no-op if the columns already exist (idempotent RunPython).
"""

from django.db import migrations, models


def _add_soft_delete_columns(apps, schema_editor):
    """Idempotently add is_deleted + deleted_at to the 3 tables.

    SQLiteStore may have already added these columns at runtime, so we
    check before ALTER to avoid duplicate-column errors.
    """
    tables = ["detection_results", "alert_records", "diagnosis_records"]
    for table in tables:
        with schema_editor.connection.cursor() as cur:
            cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
            if "is_deleted" not in cols:
                cur.execute(
                    f"ALTER TABLE {table} ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0"
                )
            if "deleted_at" not in cols:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN deleted_at REAL")


def _noop_reverse(apps, schema_editor):
    """Reverse is a no-op — we don't drop soft-delete columns on rollback."""
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("phm_site", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="detectionresult",
            name="is_deleted",
            field=models.IntegerField(db_column="is_deleted", default=0, verbose_name="已删除"),
        ),
        migrations.AddField(
            model_name="detectionresult",
            name="deleted_at",
            field=models.FloatField(blank=True, db_column="deleted_at", null=True, verbose_name="删除时间"),
        ),
        migrations.AddField(
            model_name="alertrecord",
            name="is_deleted",
            field=models.IntegerField(db_column="is_deleted", default=0, verbose_name="已删除"),
        ),
        migrations.AddField(
            model_name="alertrecord",
            name="deleted_at",
            field=models.FloatField(blank=True, db_column="deleted_at", null=True, verbose_name="删除时间"),
        ),
        migrations.AddField(
            model_name="diagnosisrecord",
            name="is_deleted",
            field=models.IntegerField(db_column="is_deleted", default=0, verbose_name="已删除"),
        ),
        migrations.AddField(
            model_name="diagnosisrecord",
            name="deleted_at",
            field=models.FloatField(blank=True, db_column="deleted_at", null=True, verbose_name="删除时间"),
        ),
        migrations.RunPython(_add_soft_delete_columns, _noop_reverse),
    ]
