"""Add raw_snapshot / score_snapshot columns to alert_records.

Stores the triggering waveform + per-sample scores so LLM diagnosis can
inspect the alert-time waveform without relying on a later telemetry
lookup (which may have scrolled past the alert point).
"""

from django.db import migrations, models


def _add_snapshot_columns(apps, schema_editor):
    """Idempotently add raw_snapshot + score_snapshot to alert_records."""
    table = "alert_records"
    with schema_editor.connection.cursor() as cur:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        if "raw_snapshot" not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN raw_snapshot TEXT")
        if "score_snapshot" not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN score_snapshot TEXT")


def _noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("phm_site", "0002_soft_delete"),
    ]

    operations = [
        migrations.AddField(
            model_name="alertrecord",
            name="raw_snapshot",
            field=models.TextField(blank=True, db_column="raw_snapshot", null=True, verbose_name="原始波形快照"),
        ),
        migrations.AddField(
            model_name="alertrecord",
            name="score_snapshot",
            field=models.TextField(blank=True, db_column="score_snapshot", null=True, verbose_name="分数快照"),
        ),
        migrations.RunPython(_add_snapshot_columns, _noop_reverse),
    ]
