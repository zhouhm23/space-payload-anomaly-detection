"""SQLite persistent storage with async batch writes.

Provides three tables:

* ``raw_telemetry``      — every ingested telemetry sample (channel, raw
                           value, space-side score, timestamps).
* ``detection_results``  — per-block three-layer cascade output (L1/L2/L3
                           decisions, scores, rules triggered, final score).
* ``alert_records``      — measured alerts + predicted warnings with
                           lifecycle status (pending/confirmed/false).

Write strategy: **double-write with async batching**.

  * RingBuffer / AlertStore / WarningStore remain the synchronous,
    in-memory, real-time stores (low latency for the frontend).
  * SQLiteStore receives data via ``enqueue_*()`` calls that simply put
    items on a ``queue.Queue``.  A background daemon thread drains the
    queue in batches (every ``batch_size`` items or every ``flush_interval``
    seconds, whichever comes first) and commits via ``executemany()``.

This keeps the poll path fast — the enqueue is O(1) and non-blocking —
while giving full persistence for historical queries.

Uses the Python standard-library ``sqlite3`` module (no new dependency).
WAL mode is enabled for concurrent read/write without blocking.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .warning_store import compute_final_status

logger = logging.getLogger(__name__)

__all__ = ["SQLiteStore"]

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "phm.db"


def _build_alert_where_clause(
    *,
    channel: str | None = None,
    alert_type: str | None = None,
    status: str | None = None,
    verdict: str | None = None,
    llm_verdict: str | None = None,
    human_verdict: str | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> tuple[str, list]:
    """Build a WHERE-clause fragment + parameter list for alert_records.

    Shared by ``query_alerts_filtered`` and ``count_alerts_filtered`` so the
    filter logic is maintained in one place. The returned sql fragment starts
    with ``AND`` (appended after ``WHERE is_deleted = 0``); params follow the
    placeholder order.
    """
    parts: list[str] = []
    params: list = []
    if channel:
        parts.append("channel = ?")
        params.append(channel)
    if alert_type:
        parts.append("alert_type = ?")
        params.append(alert_type)
    if status:
        parts.append("status = ?")
        params.append(status)
    if verdict:
        parts.append("(llm_verdict = ? OR human_verdict = ?)")
        params.extend([verdict, verdict])
    if llm_verdict:
        if llm_verdict == 'none':
            parts.append("llm_verdict IS NULL")
        else:
            parts.append("llm_verdict = ?")
            params.append(llm_verdict)
    if human_verdict:
        if human_verdict == 'none':
            parts.append("human_verdict IS NULL")
        else:
            parts.append("human_verdict = ?")
            params.append(human_verdict)
    if start_ts is not None:
        parts.append("created_at >= ?")
        params.append(start_ts)
    if end_ts is not None:
        parts.append("created_at <= ?")
        params.append(end_ts)
    if parts:
        return " AND " + " AND ".join(parts), params
    return "", params


def _build_recycle_where_clause(
    table: str,
    *,
    deleted_start: float | None = None,
    deleted_end: float | None = None,
    channel: str | None = None,
    status: str | None = None,
) -> tuple[str, list]:
    """Build a WHERE-clause fragment + params for the recycle-bin list.

    Used by ``query_deleted`` / ``count_deleted`` so the filter logic lives in
    one place. The returned fragment starts with ``AND`` (appended after
    ``WHERE is_deleted = 1``).

    Schema-aware: every admin table has ``deleted_at`` and ``channel``, so those
    two filters always apply. ``status`` only applies to ``alert_records`` —
    there it maps to the underlying ``status`` / ``llm_verdict`` /
    ``human_verdict`` columns (mirrors ``compute_final_status`` priority
    human > llm > status). On ``detection_results`` / ``diagnosis_records`` the
    status filter is silently dropped (no such concept).
    """
    parts: list[str] = []
    params: list = []
    if deleted_start is not None:
        parts.append("deleted_at >= ?")
        params.append(deleted_start)
    if deleted_end is not None:
        parts.append("deleted_at <= ?")
        params.append(deleted_end)
    if channel:
        parts.append("channel = ?")
        params.append(channel)
    if status and table == "alert_records":
        # final_status is computed (human_verdict > llm_verdict > status).
        # A row matches a given final status iff its highest-priority non-null
        # source equals it. Verification-tier values (active/pending/confirmed/
        # false) only apply when no verdict is set; verdict-tier values
        # (real/false_alarm/uncertain) match against either verdict column.
        if status in ("confirmed", "false", "pending", "active"):
            parts.append(
                "(llm_verdict IS NULL AND human_verdict IS NULL AND status = ?)"
            )
            params.append(status)
        else:  # real / false_alarm / uncertain — verdict tier
            parts.append("(human_verdict = ? OR llm_verdict = ?)")
            params.extend([status, status])
    if parts:
        return " AND " + " AND ".join(parts), params
    return "", params

_SCHEMA = """
-- Per-block three-layer cascade detection results
CREATE TABLE IF NOT EXISTS detection_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel         TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    l1_decision     TEXT,
    l1_score        REAL,
    l1_detail       TEXT,          -- JSON
    l2_score        REAL,
    l3_score        REAL,
    l3_rules        TEXT,          -- JSON array of rule names
    final_score     REAL,
    ingested_at     REAL    NOT NULL DEFAULT (unixepoch()),
    is_deleted      INTEGER NOT NULL DEFAULT 0,   -- 0=normal, 1=soft-deleted
    deleted_at      REAL                               -- soft-delete timestamp
);
CREATE INDEX IF NOT EXISTS idx_det_channel_time ON detection_results(channel, timestamp);

-- Alert + warning records (measured and predicted)
CREATE TABLE IF NOT EXISTS alert_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel     TEXT    NOT NULL,
    alert_type  TEXT    NOT NULL,   -- 'measured' or 'predicted'
    score       REAL,
    message     TEXT,
    created_at  REAL    NOT NULL,
    status      TEXT    DEFAULT 'active',   -- active / pending / confirmed / false
    verified_at REAL,
    llm_verdict  TEXT,               -- real / false_alarm / uncertain (NULL = not diagnosed)
    human_verdict TEXT,              -- real / false_alarm / uncertain (NULL = not annotated)
    raw_snapshot  TEXT,              -- JSON: raw waveform at alert time (measured alerts only)
    score_snapshot TEXT,             -- JSON: per-sample anomaly scores at alert time
    ingested_at REAL    NOT NULL DEFAULT (unixepoch()),
    is_deleted  INTEGER NOT NULL DEFAULT 0,   -- 0=normal, 1=soft-deleted
    deleted_at  REAL                               -- soft-delete timestamp
);
CREATE INDEX IF NOT EXISTS idx_alert_channel_time ON alert_records(channel, created_at);

-- LLM diagnosis results (cached per alert — one diagnosis per unique alert)
CREATE TABLE IF NOT EXISTS diagnosis_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel         TEXT    NOT NULL,
    alert_type      TEXT    NOT NULL,   -- 'measured' or 'predicted'
    alert_ts        REAL    NOT NULL,   -- the alert/warning timestamp (cache key part)
    diagnosis       TEXT,               -- Markdown report
    context_summary TEXT,               -- JSON
    elapsed_sec     REAL,
    error           TEXT,
    llm_verdict     TEXT,               -- real / false_alarm / uncertain (parsed from report)
    created_at      REAL    NOT NULL DEFAULT (unixepoch()),
    is_deleted      INTEGER NOT NULL DEFAULT 0,   -- 0=normal, 1=soft-deleted
    deleted_at      REAL                               -- soft-delete timestamp
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_diag_key ON diagnosis_records(channel, alert_type, alert_ts);
"""


class SQLiteStore:
    """Persistent SQLite store with background batch writer.

    Args:
        db_path:         path to the ``.db`` file.  Parent dirs are created.
        batch_size:      max items buffered before a flush (default 200).
        flush_interval:  max seconds between flushes (default 2.0).
        enabled:         if False, all enqueue/query calls are no-ops.
                         Useful for tests or when persistence is unwanted.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        batch_size: int = 200,
        flush_interval: float = 2.0,
        enabled: bool = True,
        sample_rate: float = 100.0,
    ) -> None:
        """Persistent SQLite store with background batch writer.

        Args:
            db_path:         path to the ``.db`` file.  Parent dirs are created.
            batch_size:      max items buffered before a flush (default 200).
            flush_interval:  max seconds between flushes (default 2.0).
            enabled:         if False, all enqueue/query calls are no-ops.
                             Useful for tests or when persistence is unwanted.
            sample_rate:     sensor sample rate in Hz.  Timestamps are
                             quantised to half the sampling interval so that
                             raw and prediction points computed via different
                             float paths collapse onto the same PRIMARY KEY,
                             while adjacent genuine samples never merge.
        """
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.enabled = enabled
        # Quantise timestamps to the sampling interval (not half).  Since pred
        # timestamps are computed as origin_q + k*sample_interval, using the
        # full interval as quantum guarantees pred lands on the EXACT same
        # grid points as raw — no odd/even grid phase mismatch.
        sample_interval = 1.0 / sample_rate if sample_rate > 0 else 0.01
        self._ts_quantum = sample_interval
        # Gap threshold: one acquisition block (512 samples @ sample_rate).
        # Gaps larger than this are treated as monitoring interruptions
        # (system shutdown / comms loss), not accidental packet drops.
        from ..config import FORECAST_CONTEXT_LENGTH
        self._gap_threshold = FORECAST_CONTEXT_LENGTH * sample_interval

        self._queue: queue.Queue[list | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()  # guards _conn writes
        self._created_tables: set[str] = set()  # cache of per-channel tables
        # Flush counter for periodic WAL checkpointing (see _flush_loop).
        # Reset to 0 whenever the writer thread (re)starts.
        self._flush_count = 0

        if self.enabled:
            self._init_db()

    def _quantize_ts(self, ts: float) -> float:
        """Snap a timestamp to the nearest quantum grid point."""
        if self._ts_quantum <= 0:
            return float(ts)
        return float(math.floor(ts / self._ts_quantum + 0.5) * self._ts_quantum)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # WAL tuning (v1.2 perf): the default wal_autocheckpoint (1000 pages,
        # ~4 MB) was being outrun by the eval-thread write rate, letting the
        # WAL balloon to ~900 MB in production. A 900 MB WAL forces every new
        # reader connection (each Django request) to search a huge WAL index,
        # amplifying read latency. Lower the threshold to 256 pages (~1 MB)
        # so checkpoints fire far more frequently and the WAL stays small.
        # PASSIVE mode (the default) is used so writers are never blocked.
        self._conn.execute("PRAGMA wal_autocheckpoint=256")
        self._conn.executescript(_SCHEMA)
        self._migrate_verdict_columns()
        self._migrate_soft_delete_columns()
        self._migrate_snapshot_columns()
        # Per-channel telemetry tables are created lazily by _ensure_tel_table
        # (which already includes deleted_at + origin in its DDL). This pass
        # upgrades pre-existing telemetry_* tables from older DBs that were
        # created before the soft-delete / origin feature.
        self._migrate_tel_soft_delete_columns()
        logger.info("SQLiteStore initialised: %s", self.db_path)

    def _migrate_verdict_columns(self) -> None:
        """Add llm_verdict/human_verdict to alert_records and llm_verdict
        to diagnosis_records if they don't exist (backward compat with
        pre-existing DBs created before the verdict feature)."""
        if self._conn is None:
            return
        for table, col in [
            ("alert_records", "llm_verdict"),
            ("alert_records", "human_verdict"),
            ("diagnosis_records", "llm_verdict"),
        ]:
            cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if col not in cols:
                try:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
                    logger.info("Migrated %s: added column %s", table, col)
                except Exception:
                    logger.warning("Failed to add column %s to %s", col, table, exc_info=True)

    def _migrate_soft_delete_columns(self) -> None:
        """Add is_deleted/deleted_at to the three business tables if they
        don't exist (backward compat with pre-existing DBs created before
        the soft-delete feature)."""
        if self._conn is None:
            return
        for table in ("detection_results", "alert_records", "diagnosis_records"):
            cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if "is_deleted" not in cols:
                try:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0"
                    )
                    logger.info("Migrated %s: added column is_deleted", table)
                except Exception:
                    logger.warning("Failed to add is_deleted to %s", table, exc_info=True)
            if "deleted_at" not in cols:
                try:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN deleted_at REAL")
                    logger.info("Migrated %s: added column deleted_at", table)
                except Exception:
                    logger.warning("Failed to add deleted_at to %s", table, exc_info=True)

    def _migrate_snapshot_columns(self) -> None:
        """Add raw_snapshot/score_snapshot to alert_records if they don't
        exist (backward compat with pre-existing DBs created before the
        alert-snapshot feature)."""
        if self._conn is None:
            return
        for col in ("raw_snapshot", "score_snapshot"):
            cols = [r[1] for r in self._conn.execute("PRAGMA table_info(alert_records)").fetchall()]
            if col not in cols:
                try:
                    self._conn.execute(f"ALTER TABLE alert_records ADD COLUMN {col} TEXT")
                    logger.info("Migrated alert_records: added column %s", col)
                except Exception:
                    logger.warning("Failed to add %s to alert_records", col, exc_info=True)

    def _migrate_tel_soft_delete_columns(self) -> None:
        """Add ``deleted_at`` and ``origin`` columns to every existing
        ``telemetry_*`` table (lazy, idempotent).

        Per-channel telemetry tables are created lazily by
        ``_ensure_tel_table`` (whose DDL already includes these columns), so
        tables created on this code version already have them. This pass
        upgrades pre-existing tables from older DBs that pre-date the
        soft-delete / origin-tagging feature:

          * ``deleted_at REAL``         — soft-delete timestamp (NULL = alive)
          * ``origin TEXT DEFAULT 'acq'``— 'acq' for acquisition rows,
                                             'manual' for backfilled rows

        Both columns are optional (nullable / default-filled) so existing
        queries and writes keep working unchanged.
        """
        if not self.enabled or self._conn is None:
            return
        tables = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'telemetry_%'"
        ).fetchall()
        for (tbl,) in tables:
            cols = {r[1] for r in self._conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()}
            if "deleted_at" not in cols:
                try:
                    self._conn.execute(f'ALTER TABLE "{tbl}" ADD COLUMN deleted_at REAL')
                    self._conn.execute(
                        f'CREATE INDEX IF NOT EXISTS "idx_{tbl}_del" ON "{tbl}"(deleted_at)'
                    )
                    logger.info("Migrated %s: added column deleted_at", tbl)
                except Exception:
                    logger.warning("Failed to add deleted_at to %s", tbl, exc_info=True)
            if "origin" not in cols:
                try:
                    # NOT NULL DEFAULT 'acq': every acquisition row is tagged
                    # automatically, matching what _ensure_tel_table's DDL does
                    # for freshly-created tables.
                    self._conn.execute(
                        f"ALTER TABLE \"{tbl}\" ADD COLUMN origin TEXT NOT NULL DEFAULT 'acq'"
                    )
                    logger.info("Migrated %s: added column origin", tbl)
                except Exception:
                    logger.warning("Failed to add origin to %s", tbl, exc_info=True)

    # Per-channel telemetry tables: each channel gets its own table
    # (e.g. ``telemetry_C_1``, ``telemetry_VS_multi_sine``) so channels
    # are fully isolated.  Table names are derived from the channel name
    # by replacing non-alphanumeric characters with underscores.
    _created_tables: set[str] = None  # set per-instance in __init__

    @staticmethod
    def _tel_table(channel: str) -> str:
        """Convert a channel name to a safe telemetry table name."""
        safe = "".join(c if c.isalnum() else "_" for c in channel)
        return f"telemetry_{safe}"

    def _ensure_tel_table(self, channel: str) -> str:
        """Create the per-channel telemetry table if it doesn't exist yet."""
        table = self._tel_table(channel)
        if self._created_tables is not None and table in self._created_tables:
            return table
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
                timestamp                   REAL    NOT NULL PRIMARY KEY,
                raw_value                   REAL,
                anomaly_score               REAL,
                predicted_value             REAL,
                predicted_anomaly_score     REAL,
                origin_ts                   REAL,
                ingested_at                 REAL    NOT NULL DEFAULT (unixepoch()),
                deleted_at                  REAL,                          -- soft-delete ts (NULL = alive)
                origin                      TEXT    NOT NULL DEFAULT 'acq'  -- 'acq' | 'manual'
            )
        """)
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table}_ts" ON "{table}"(timestamp)'
        )
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table}_del" ON "{table}"(deleted_at)'
        )
        if self._created_tables is not None:
            self._created_tables.add(table)
        return table

    def start(self) -> None:
        """Start the background flush thread."""
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="SQLiteStore"
        )
        self._thread.start()
        logger.debug("SQLiteStore flush thread started")

    def close(self, timeout: float = 5.0) -> None:
        """Signal the flush thread to drain and stop, then close the DB."""
        if not self.enabled:
            return
        self._stop_event.set()
        self._queue.put(None)  # wake up the loop
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        # Drain anything still in the queue (the thread may have exited
        # before processing all items if timeout was short).
        self._flush_remaining()
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        logger.debug("SQLiteStore closed")

    # ------------------------------------------------------------------
    # Producer API (non-blocking, called from poll path)
    # ------------------------------------------------------------------

    def enqueue_telemetry(
        self,
        channel: str,
        raw: float,
        score: float | None,
        received_at: float,
    ) -> None:
        if not self.enabled:
            return
        self._queue.put(["raw", channel, float(raw),
                         float(score) if score is not None else None,
                         float(received_at)])

    def enqueue_telemetry_batch(self, entries: Iterable[dict]) -> None:
        """Enqueue multiple telemetry entries from a poll result.

        Each entry must have keys: ``channel``, ``raw``, ``score``, ``received_at``.
        Puts all items in a single ``queue.put`` to minimise lock contention.

        Timestamps are quantised to the sensor sampling grid so raw and
        prediction rows computed via different float paths collapse onto
        the same PRIMARY KEY (otherwise a ~1ms offset splits them into two
        rows and the predicted curve breaks up).
        """
        if not self.enabled:
            return
        batch: list = []
        for e in entries:
            batch.append([
                "raw", e["channel"], float(e["raw"]),
                float(e["score"]) if e.get("score") is not None else None,
                self._quantize_ts(e["received_at"]),
            ])
        if batch:
            self._queue.put(batch)

    def enqueue_detection(
        self,
        channel: str,
        timestamp: float,
        cascade_output: Any,
    ) -> None:
        """Enqueue a :class:`CascadeOutput` (or dict with equivalent shape).

        Extracts L1/L2/L3 decisions, scores, rules and final score.
        """
        if not self.enabled:
            return

        l1_decision = l1_score = None
        l1_detail: dict = {}
        l2_score = l3_score = final_score = None
        l3_rules: list = []

        # Support both CascadeOutput objects and plain dicts
        layers = getattr(cascade_output, "layers", None)
        if layers is None and isinstance(cascade_output, dict):
            layers = cascade_output.get("layers", [])
        if layers is None:
            layers = []

        for lr in layers:
            layer = lr.layer if hasattr(lr, "layer") else lr.get("layer", "")
            decision = lr.decision if hasattr(lr, "decision") else lr.get("decision", "")
            score = lr.score if hasattr(lr, "score") else lr.get("score", 0.0)
            detail = lr.detail if hasattr(lr, "detail") else lr.get("detail", {})

            if layer == "L1_classic":
                l1_decision = decision
                l1_score = score
                l1_detail = {k: v for k, v in detail.items()
                             if k != "per_sample_score"}
            elif layer == "L2_dl":
                l2_score = score
            elif layer == "L3_physical":
                l3_score = score
                rules = detail.get("rules", [])
                l3_rules = rules if isinstance(rules, list) else [str(rules)]

        if hasattr(cascade_output, "final_scores") and cascade_output.final_scores is not None:
            fs = cascade_output.final_scores
            final_score = float(np.nanmax(fs)) if len(fs) > 0 else 0.0
        elif isinstance(cascade_output, dict) and "final_scores" in cascade_output:
            fs = cascade_output["final_scores"]
            final_score = float(np.nanmax(fs)) if len(fs) > 0 else 0.0

        self._queue.put([
            "det",
            channel,
            float(timestamp),
            l1_decision,
            l1_score,
            json.dumps(l1_detail, ensure_ascii=False, default=str),
            l2_score,
            l3_score,
            json.dumps(l3_rules, ensure_ascii=False),
            final_score,
        ])

    def enqueue_alert(self, alert: dict) -> None:
        """Enqueue an alert/warning record.

        Expected keys: ``channel``, ``type`` (or ``alert_type``), ``score``,
        ``message``, ``time`` (or ``created_at``), ``status``,
        ``raw_snapshot`` (list, measured alerts), ``score_snapshot`` (list).
        """
        if not self.enabled:
            return
        alert_type = alert.get("alert_type") or alert.get("type", "measured")
        created = alert.get("created_at") or alert.get("time", time.time())
        # Serialize snapshots to JSON strings now (before they hit the
        # background flush thread, which doesn't know about list→TEXT).
        raw_snap = alert.get("raw_snapshot")
        score_snap = alert.get("score_snapshot")
        raw_json = json.dumps(raw_snap) if raw_snap is not None else None
        score_json = json.dumps(score_snap) if score_snap is not None else None
        self._queue.put([
            "alert",
            alert.get("channel", ""),
            alert_type,
            alert.get("score"),
            alert.get("message", ""),
            float(created),
            alert.get("status", "active"),
            alert.get("verified_at"),
            raw_json,
            score_json,
        ])

    # ------------------------------------------------------------------
    # Consumer: background flush loop
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # Block until at least one item arrives (or timeout)
                item = self._queue.get(timeout=self.flush_interval)
                self._queue.task_done()
            except queue.Empty:
                continue
            if item is None:
                # Sentinel from close() — drain remaining and exit
                self._flush_remaining()
                break
            # Collect this item + drain the rest without blocking
            batch = [item]
            while True:
                try:
                    extra = self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
                if extra is not None:
                    batch.append(extra)
            self._write_batch(batch)
            # Periodic WAL checkpoint (v1.2 perf). The background writer keeps
            # inserting into 5 multi-million-row telemetry tables; without an
            # active checkpoint the WAL grew to ~900 MB in production and every
            # Django reader request paid a huge WAL-index search cost. PASSIVE
            # mode never blocks writers — it just merges whatever WAL frames
            # are safe to fold back into the main db. Firing every ~15 flushes
            # (≈30 s at the default 2 s interval) keeps the WAL small without
            # adding measurable overhead.
            self._flush_count += 1
            if self._flush_count % 15 == 0:
                self._wal_checkpoint()

    def _wal_checkpoint(self) -> None:
        """Run a PASSIVE WAL checkpoint (best-effort, never blocks writers).

        Called periodically from the flush thread. Failures are logged at
        debug level only — a missed checkpoint is not fatal, the WAL will be
        folded on the next attempt or on connection close.
        """
        if not self.enabled or self._conn is None:
            return
        try:
            with self._write_lock:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception:
            logger.debug("wal_checkpoint failed (non-fatal)", exc_info=True)

    def _flush_remaining(self) -> None:
        """Drain all pending items and commit in batches (used by close)."""
        if self._conn is None:
            return
        items: list = []
        while True:
            try:
                item = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
            if item is not None:
                items.append(item)
        if items:
            self._write_batch(items)

    def _write_batch(self, items: list) -> None:
        """Write a batch of items to the database.

        Each item in *items* may be either:
          - a single record list ``["raw", ch, raw, score, ts]``
          - a nested batch (list of record lists) produced by
            ``enqueue_telemetry_batch``
        We flatten both into a flat list of records before processing.
        """
        if self._conn is None or not items:
            return

        # Flatten: if an item's first element is a list, it's a nested batch
        flat: list = []
        for it in items:
            if isinstance(it, list) and len(it) > 0 and isinstance(it[0], list):
                flat.extend(it)
            else:
                flat.append(it)

        raw_rows = [
            (r[1], r[4], r[2], r[3])   # (channel, timestamp=received_at, raw_value=raw, anomaly_score=score)
            for r in flat if r[0] == "raw"
        ]
        det_rows = [
            (r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9])
            for r in flat if r[0] == "det"
        ]
        alert_rows = [
            (r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9])
            for r in flat if r[0] == "alert"
        ]
        # Predictions are now individual UPSERTs into the unified telemetry
        # table (one row per predicted point), not a batch JSON blob.
        pred_rows = [
            (r[1], r[2], r[3], r[4], r[5])  # channel, ts, pred_val, pred_score, origin_ts
            for r in flat if r[0] == "pred"
        ]

        try:
            with self._write_lock:
                # Raw rows: group by channel, each channel has its own table.
                # raw_rows format: (channel, timestamp, raw_value, anomaly_score)
                if raw_rows:
                    by_ch: dict[str, list] = {}
                    for row in raw_rows:
                        by_ch.setdefault(row[0], []).append(row[1:])  # drop channel, keep (ts, raw, score)
                    for ch, rows in by_ch.items():
                        table = self._ensure_tel_table(ch)
                        self._conn.executemany(
                            f'INSERT INTO "{table}" (timestamp, raw_value, anomaly_score) '
                            f'VALUES (?,?,?) '
                            f'ON CONFLICT(timestamp) DO UPDATE SET '
                            f'raw_value = excluded.raw_value, '
                            f'anomaly_score = excluded.anomaly_score',
                            rows,
                        )
                if det_rows:
                    self._conn.executemany(
                        """INSERT INTO detection_results
                           (channel, timestamp, l1_decision, l1_score, l1_detail,
                            l2_score, l3_score, l3_rules, final_score, ingested_at, is_deleted)
                           VALUES (?,?,?,?,?,?,?,?,?, unixepoch(), 0)""",
                        det_rows,
                    )
                if alert_rows:
                    self._conn.executemany(
                        """INSERT INTO alert_records
                           (channel, alert_type, score, message, created_at, status, verified_at,
                            raw_snapshot, score_snapshot, ingested_at, is_deleted)
                           VALUES (?,?,?,?,?,?,?,?,?, unixepoch(), 0)""",
                        alert_rows,
                    )
                # Pred rows: group by channel, same per-channel table.
                # pred_rows format: (channel, timestamp, pred_val, pred_score, origin_ts)
                if pred_rows:
                    by_ch_p: dict[str, list] = {}
                    for row in pred_rows:
                        by_ch_p.setdefault(row[0], []).append(row[1:])  # drop channel
                    for ch, rows in by_ch_p.items():
                        table = self._ensure_tel_table(ch)
                        self._conn.executemany(
                            f'INSERT INTO "{table}" '
                            f'(timestamp, predicted_value, predicted_anomaly_score, origin_ts) '
                            f'VALUES (?,?,?,?) '
                            f'ON CONFLICT(timestamp) DO UPDATE SET '
                            f'predicted_value = excluded.predicted_value, '
                            f'predicted_anomaly_score = excluded.predicted_anomaly_score, '
                            f'origin_ts = excluded.origin_ts',
                            rows,
                        )
        except Exception:
            logger.warning("SQLite batch flush failed (%d items)", len(items), exc_info=True)

    # ------------------------------------------------------------------
    # Predictions (predicted values + predicted anomaly scores)
    # ------------------------------------------------------------------

    def enqueue_predictions(
        self,
        channel: str,
        origin_ts: float,
        predict_start: float,
        predict_end: float,
        prediction: list[float],
        predict_scores: list[float] | None = None,
        model: str | None = None,
        timestamps: list[float] | None = None,
    ) -> None:
        """Enqueue predicted values as individual UPSERT rows.

        Each predicted point becomes a separate row in the unified
        ``telemetry`` table, keyed by (channel, timestamp).

        If *timestamps* is provided, those exact timestamps are used
        (quantised) — this avoids recomputing timestamps via a different
        float path than the raw data, which previously caused raw and pred
        to land on different PRIMARY KEYs.  When *timestamps* is None, falls
        back to linear interpolation between predict_start and predict_end.
        """
        if not self.enabled:
            return
        n = len(prediction)
        if n == 0:
            return
        scores = predict_scores if predict_scores is not None else [None] * n
        # Pred timestamps must land on the EXACT same quantum grid as raw.
        # The raw grid is origin_q + k * sample_interval (k=1,2,...), where
        # sample_interval = 1/sample_rate (the true sensor cadence) and
        # origin_q is the quantised last raw timestamp.  This guarantees
        # UPSERT merges pred into the same row as the future raw sample.
        #
        # _ts_quantum is already 1/sample_rate (see __init__), so use it
        # directly.  A previous version multiplied by 2 here (a leftover
        # from when _ts_quantum was half the sampling interval) which made
        # the pred step TWICE the raw step — pred landed on .65/.67/.69
        # while raw lived on every position, leaving "holes" (.64/.66/...)
        # with no row at all on the chart timeline.
        origin_q = self._quantize_ts(origin_ts)
        sample_interval = self._ts_quantum
        for i in range(n):
            ts = self._quantize_ts(origin_q + (i + 1) * sample_interval)
            self._queue.put([
                "pred",
                channel,
                ts,
                float(prediction[i]) if prediction[i] is not None else None,
                float(scores[i]) if i < len(scores) and scores[i] is not None else None,
                float(origin_ts),
            ])

    # ------------------------------------------------------------------
    # Window query — unified telemetry (raw + prediction in same rows)
    # ------------------------------------------------------------------

    def query_window(
        self,
        channel: str,
        count: int = 512,
        end_ts: float | None = None,
    ) -> dict:
        """Return the latest ``count`` rows from the per-channel telemetry
        table.  Each row may have raw_value and/or predicted_value filled.

        Returns dict::

            {"channel": ..., "count": N, "end_ts": ..., "start_ts": ...,
             "data": [{timestamp, raw_value, anomaly_score,
                       predicted_value, predicted_anomaly_score}, ...]}
        """
        if not self.enabled or self._conn is None:
            return {"channel": channel, "count": 0, "data": []}

        table = self._ensure_tel_table(channel)
        try:
            if end_ts is None:
                row = self._conn.execute(
                    f'SELECT MAX(timestamp) FROM "{table}"'
                ).fetchone()
                end_ts = row[0] if row and row[0] is not None else None

            if end_ts is None:
                return {"channel": channel, "count": 0, "end_ts": None,
                        "start_ts": None, "data": []}

            cur = self._conn.execute(
                f'SELECT timestamp, raw_value, anomaly_score, '
                f'predicted_value, predicted_anomaly_score '
                f'FROM "{table}" WHERE timestamp <= ? '
                f'ORDER BY timestamp DESC LIMIT ?',
                [end_ts, count],
            )
            rows = list(reversed(cur.fetchall()))

            # Deduplicate by timestamp (1ms granularity)
            seen: set = set()
            deduped: list = []
            for r in rows:
                ts = round(r[0], 3)
                if ts in seen:
                    continue
                seen.add(ts)
                deduped.append(r)

            start_ts = deduped[0][0] if deduped else end_ts

            # Detect monitoring gaps: consecutive rows whose timestamp
            # difference exceeds one acquisition block (system shutdown /
            # comms interruption).  Accidental single-point drops are NOT
            # flagged (they are far shorter than a block).
            gaps: list[dict] = []
            for i in range(1, len(deduped)):
                dt = deduped[i][0] - deduped[i - 1][0]
                if dt > self._gap_threshold:
                    gaps.append({
                        "start": deduped[i - 1][0],
                        "end": deduped[i][0],
                        "duration": dt,
                        "index": i,  # position in data[] where the gap ends
                    })

            data = [
                {
                    "timestamp": r[0],
                    "raw_value": r[1],
                    "anomaly_score": r[2],
                    "predicted_value": r[3],
                    "predicted_anomaly_score": r[4],
                }
                for r in deduped
            ]

            return {
                "channel": channel,
                "count": len(data),
                "end_ts": end_ts,
                "start_ts": start_ts,
                "gap_threshold": self._gap_threshold,
                "gaps": gaps,
                "data": data,
            }
        except Exception:
            logger.warning("query_window failed", exc_info=True)
            return {"channel": channel, "count": 0, "data": [], "gaps": []}

    # ------------------------------------------------------------------
    # Admin pagination: per-channel telemetry tables (v1.2 admin page)
    # ------------------------------------------------------------------
    #
    # Telemetry tables have no AUTOINCREMENT ``id`` column (timestamp is the
    # PRIMARY KEY), so these methods use the implicit SQLite ``rowid`` as the
    # operation identifier. rowid is per-table, so callers must always pair
    # it with the channel name.
    #
    # Soft-delete on telemetry uses ``deleted_at`` (NULL = alive), matching
    # the lazy migration in ``_migrate_tel_soft_delete_columns``.

    @staticmethod
    def _build_tel_where_clause(
        *,
        start_ts: float | None,
        end_ts: float | None,
        include_deleted: bool,
        value_type: str | None = None,
    ) -> tuple[str, list]:
        """Build a WHERE-clause fragment + params for a telemetry_* table.

        Shared by ``query_tel_page`` and ``count_tel`` so the filter logic
        lives in one place. The returned fragment starts with ``WHERE``
        (no leading ``AND``); params follow the placeholder order.

        ``value_type`` optionally restricts rows by which value columns are
        non-null: ``"raw"`` → ``raw_value IS NOT NULL``, ``"predicted"`` →
        ``predicted_value IS NOT NULL``, ``"both"`` → both non-null.
        ``None`` / ``"all"`` / any other value applies no value filter.
        """
        parts: list[str] = []
        params: list = []
        if not include_deleted:
            parts.append("deleted_at IS NULL")
        if start_ts is not None:
            parts.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts is not None:
            parts.append("timestamp <= ?")
            params.append(end_ts)
        if value_type == "raw":
            parts.append("raw_value IS NOT NULL")
        elif value_type == "predicted":
            parts.append("predicted_value IS NOT NULL")
        elif value_type == "both":
            parts.append("raw_value IS NOT NULL")
            parts.append("predicted_value IS NOT NULL")
        if parts:
            return "WHERE " + " AND ".join(parts), params
        return "", params

    def query_tel_page(
        self,
        channel: str,
        *,
        offset: int = 0,
        limit: int = 20,
        start_ts: float | None = None,
        end_ts: float | None = None,
        include_deleted: bool = False,
        value_type: str | None = None,
    ) -> list[dict]:
        """Paginated telemetry query for the admin data-management page.

        Returns rows newest-first (ORDER BY timestamp DESC, consistent with
        ``query_alerts_filtered``). Each row dict carries the implicit rowid
        (as ``id``) plus all telemetry columns so the front end can drive
        edit / delete / export actions:

            {id, timestamp, raw_value, anomaly_score, predicted_value,
             predicted_anomaly_score, origin_ts, ingested_at, deleted_at,
             origin}

        Args:
            channel:        per-channel table to query.
            offset:         paging offset (rows skipped, newest first).
            limit:          page size (clamped to 1..1000).
            start_ts:       optional inclusive lower timestamp bound.
            end_ts:         optional inclusive upper timestamp bound.
            include_deleted: if True, also return soft-deleted rows
                             (``deleted_at IS NOT NULL``).
            value_type:     optional value-column filter
                            (``"raw"`` / ``"predicted"`` / ``"both"``);
                            ``None`` or ``"all"`` applies no filter.

        Returns ``[]`` on failure or empty channel.
        """
        if not self.enabled or self._conn is None:
            return []
        try:
            limit = max(1, min(int(limit), 1000))
        except (TypeError, ValueError):
            limit = 20
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        table = self._ensure_tel_table(channel)
        where_sql, params = self._build_tel_where_clause(
            start_ts=start_ts, end_ts=end_ts, include_deleted=include_deleted,
            value_type=value_type,
        )
        sql = (
            f'SELECT rowid AS id, timestamp, raw_value, anomaly_score, '
            f'predicted_value, predicted_anomaly_score, origin_ts, '
            f'ingested_at, deleted_at, origin '
            f'FROM "{table}" {where_sql} '
            f'ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        )
        params.append(limit)
        params.append(offset)
        try:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        except Exception:
            logger.warning("query_tel_page(%s) failed", channel, exc_info=True)
            return []
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "raw_value": r[2],
                "anomaly_score": r[3],
                "predicted_value": r[4],
                "predicted_anomaly_score": r[5],
                "origin_ts": r[6],
                "ingested_at": r[7],
                "deleted_at": r[8],
                "origin": r[9],
            }
            for r in rows
        ]

    def count_tel(
        self,
        channel: str,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        include_deleted: bool = False,
        value_type: str | None = None,
    ) -> int:
        """Total row count of a telemetry table for pagination metadata.

        Filter params mirror ``query_tel_page`` (without limit/offset),
        including the optional ``value_type`` value-column filter
        (``"raw"`` / ``"predicted"`` / ``"both"``; ``None`` or ``"all"``
        applies no filter). Returns 0 on failure or empty channel.
        """
        if not self.enabled or self._conn is None:
            return 0
        table = self._ensure_tel_table(channel)
        where_sql, params = self._build_tel_where_clause(
            start_ts=start_ts, end_ts=end_ts, include_deleted=include_deleted,
            value_type=value_type,
        )
        sql = f'SELECT COUNT(*) FROM "{table}" {where_sql}'
        try:
            cur = self._conn.execute(sql, params)
            return int(cur.fetchone()[0])
        except Exception:
            logger.warning("count_tel(%s) failed", channel, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Query API (synchronous, for /api/history and /api/detection)
    # ------------------------------------------------------------------

    def query_history(
        self,
        channel: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Query telemetry history (raw_value only; NULLs excluded).

        If *channel* is given, queries that channel's per-channel table.
        If None, queries all telemetry_* tables via UNION.
        """
        if not self.enabled or self._conn is None:
            return []
        try:
            if channel:
                table = self._ensure_tel_table(channel)
                sql = (f"SELECT '{channel}' AS channel, timestamp, raw_value, anomaly_score "
                       f'FROM "{table}" WHERE raw_value IS NOT NULL')
                params: list = []
                if start_time is not None:
                    sql += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time is not None:
                    sql += " AND timestamp <= ?"
                    params.append(end_time)
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                cur = self._conn.execute(sql, params)
                rows = cur.fetchall()
            else:
                # No channel: query all telemetry_* tables via UNION.
                # Derive channel name from table name (e.g. telemetry_C_1 → C-1).
                tables = self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'telemetry_%'"
                ).fetchall()
                if not tables:
                    return []
                unions = []
                for (t,) in tables:
                    ch_name = t[len("telemetry_"):]
                    unions.append(f"SELECT '{ch_name}' AS channel, timestamp, raw_value, anomaly_score "
                                  f'FROM "{t}" WHERE raw_value IS NOT NULL')
                sql = " UNION ALL ".join(unions) + " ORDER BY timestamp DESC LIMIT ?"
                cur = self._conn.execute(sql, [limit])
                rows = cur.fetchall()
        except Exception:
            logger.warning("query_history failed", exc_info=True)
            return []
        return [
            {"channel": r[0], "received_at": r[1], "raw": r[2], "score": r[3]}
            for r in reversed(rows)  # chronological order
        ]

    def query_detection(
        self,
        channel: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query three-layer detection results."""
        if not self.enabled or self._conn is None:
            return []
        sql = (
            "SELECT channel, timestamp, l1_decision, l1_score, l1_detail, "
            "l2_score, l3_score, l3_rules, final_score "
            "FROM detection_results WHERE is_deleted = 0"
        )
        params: list = []
        if channel:
            sql += " AND channel = ?"
            params.append(channel)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        try:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        except Exception:
            logger.warning("query_detection failed", exc_info=True)
            return []
        results = []
        for r in reversed(rows):
            try:
                l1_detail = json.loads(r[4]) if r[4] else {}
            except Exception:
                l1_detail = {}
            try:
                l3_rules = json.loads(r[7]) if r[7] else []
            except Exception:
                l3_rules = []
            results.append({
                "channel": r[0],
                "timestamp": r[1],
                "l1_decision": r[2],
                "l1_score": r[3],
                "l1_detail": l1_detail,
                "l2_score": r[5],
                "l3_score": r[6],
                "l3_rules": l3_rules,
                "final_score": r[8],
            })
        return results

    def query_alerts(self, limit: int = 50) -> list[dict]:
        """Query persisted alert records (with ``id`` for PATCH support)."""
        if not self.enabled or self._conn is None:
            return []
        sql = (
            "SELECT id, channel, alert_type, score, message, created_at, status, verified_at, "
            "llm_verdict, human_verdict, raw_snapshot, score_snapshot "
            "FROM alert_records WHERE is_deleted = 0 ORDER BY created_at DESC LIMIT ?"
        )
        try:
            cur = self._conn.execute(sql, [limit])
            rows = cur.fetchall()
        except Exception:
            logger.warning("query_alerts failed", exc_info=True)
            return []
        results = []
        for r in reversed(rows):
            llm_v = r[8]
            human_v = r[9]
            try:
                raw_snap = json.loads(r[10]) if r[10] else None
            except (json.JSONDecodeError, TypeError):
                raw_snap = None
            try:
                score_snap = json.loads(r[11]) if r[11] else None
            except (json.JSONDecodeError, TypeError):
                score_snap = None
            results.append({
                "id": r[0],
                "channel": r[1],
                "alert_type": r[2],
                "score": r[3],
                "message": r[4],
                "created_at": r[5],
                "status": r[6],
                "verified_at": r[7],
                "llm_verdict": llm_v,
                "human_verdict": human_v,
                "raw_snapshot": raw_snap,
                "score_snapshot": score_snap,
                "final_status": compute_final_status(r[6], llm_v, human_v),
            })
        return results

    # ------------------------------------------------------------------
    # Diagnosis cache (synchronous, for POST /api/diagnosis)
    # ------------------------------------------------------------------

    def get_diagnosis(self, channel: str, alert_type: str, alert_ts: float) -> dict | None:
        """Return cached diagnosis for (channel, alert_type, alert_ts) or None."""
        if not self.enabled or self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT diagnosis, context_summary, elapsed_sec, error, created_at, llm_verdict "
                "FROM diagnosis_records WHERE is_deleted = 0 "
                "AND channel=? AND alert_type=? AND alert_ts=?",
                [channel, alert_type, alert_ts],
            ).fetchone()
            if row is None:
                return None
            import json as _json
            return {
                "diagnosis": row[0],
                "context_summary": _json.loads(row[1]) if row[1] else {},
                "elapsed_sec": row[2],
                "error": row[3],
                "created_at": row[4],
                "llm_verdict": row[5],
            }
        except Exception:
            logger.warning("get_diagnosis failed", exc_info=True)
            return None

    def save_diagnosis(self, channel: str, alert_type: str, alert_ts: float,
                       diagnosis: str, context_summary: dict,
                       elapsed_sec: float, error: str | None,
                       verdict: str | None = None) -> None:
        """Insert or replace a diagnosis record (keyed by channel+type+alert_ts)."""
        if not self.enabled or self._conn is None:
            return
        import json as _json
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO diagnosis_records "
                "(channel, alert_type, alert_ts, diagnosis, context_summary, "
                " elapsed_sec, error, llm_verdict, created_at, is_deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, unixepoch(), 0)",
                [channel, alert_type, alert_ts, diagnosis,
                 _json.dumps(context_summary, ensure_ascii=False),
                 elapsed_sec, error, verdict],
            )
            self._conn.commit()
        except Exception:
            logger.warning("save_diagnosis failed", exc_info=True)

    def list_diagnosis_keys(self, limit: int = 200) -> list[dict]:
        """Return cached diagnosis keys (channel, alert_type, alert_ts, llm_verdict)."""
        if not self.enabled or self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                "SELECT channel, alert_type, alert_ts, llm_verdict FROM diagnosis_records "
                "WHERE is_deleted = 0 ORDER BY created_at DESC LIMIT ?",
                [limit],
            )
            return [
                {"channel": r[0], "alert_type": r[1], "alert_ts": r[2], "llm_verdict": r[3]}
                for r in cur.fetchall()
            ]
        except Exception:
            logger.warning("list_diagnosis_keys failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Mutation API (synchronous, for DELETE / PATCH endpoints)
    # ------------------------------------------------------------------

    def delete_history(
        self,
        channel: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> int:
        """Delete telemetry rows matching the filter.

        If *channel* is given, deletes from that channel's per-channel
        table.  If None, deletes from ALL telemetry_* tables.

        Note: telemetry uses HARD delete (unlike alert/detection/diagnosis
        which are soft-deleted).  Telemetry is high-volume rolling data
        with no audit value — soft-deleting would make tables grow forever.
        """
        if not self.enabled or self._conn is None:
            return 0
        try:
            with self._write_lock:
                if channel:
                    table = self._ensure_tel_table(channel)
                    sql = f'DELETE FROM "{table}" WHERE 1=1'
                    params: list = []
                    if start_time is not None:
                        sql += " AND timestamp >= ?"
                        params.append(start_time)
                    if end_time is not None:
                        sql += " AND timestamp <= ?"
                        params.append(end_time)
                    cur = self._conn.execute(sql, params)
                    return cur.rowcount if cur.rowcount is not None else 0
                else:
                    # Delete from all telemetry_* tables
                    tables = self._conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'telemetry_%'"
                    ).fetchall()
                    total = 0
                    for (t,) in tables:
                        sql = f'DELETE FROM "{t}" WHERE 1=1'
                        params = []
                        if start_time is not None:
                            sql += " AND timestamp >= ?"
                            params.append(start_time)
                        if end_time is not None:
                            sql += " AND timestamp <= ?"
                            params.append(end_time)
                        cur = self._conn.execute(sql, params)
                        total += cur.rowcount if cur.rowcount is not None else 0
                    return total
        except Exception:
            logger.warning("delete_history failed", exc_info=True)
            return 0

    def delete_detection(
        self,
        channel: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> int:
        """Soft-delete detection-result rows matching the filter.

        Marks rows as ``is_deleted=1`` (does not physically remove them).
        Note the time column is ``timestamp`` here, not ``received_at``.
        """
        return self._soft_delete(
            "detection_results", "timestamp",
            channel=channel, start_time=start_time, end_time=end_time,
        )

    def delete_alert(
        self,
        channel: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> int:
        """Soft-delete alert records matching the filter.

        Marks rows as ``is_deleted=1``.  Time column is ``created_at``.
        """
        return self._soft_delete(
            "alert_records", "created_at",
            channel=channel, start_time=start_time, end_time=end_time,
        )

    def delete_diagnosis(
        self,
        channel: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> int:
        """Soft-delete diagnosis records matching the filter.

        Marks rows as ``is_deleted=1``.  Time column is ``alert_ts``.
        """
        return self._soft_delete(
            "diagnosis_records", "alert_ts",
            channel=channel, start_time=start_time, end_time=end_time,
        )

    def _soft_delete(
        self,
        table: str,
        time_col: str,
        *,
        channel: str | None,
        start_time: float | None,
        end_time: float | None,
    ) -> int:
        """Shared soft-delete builder: marks matching rows ``is_deleted=1``."""
        if not self.enabled or self._conn is None:
            return 0
        sql = f"UPDATE {table} SET is_deleted = 1, deleted_at = unixepoch() WHERE is_deleted = 0"
        params: list = []
        if channel:
            sql += " AND channel = ?"
            params.append(channel)
        if start_time is not None:
            sql += f" AND {time_col} >= ?"
            params.append(start_time)
        if end_time is not None:
            sql += f" AND {time_col} <= ?"
            params.append(end_time)
        try:
            with self._write_lock:
                cur = self._conn.execute(sql, params)
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("soft-delete from %s failed", table, exc_info=True)
            return 0

    def purge_deleted(self, table: str, older_than: float | None = None) -> int:
        """Physically remove soft-deleted rows (admin maintenance).

        Args:
            table: one of ``detection_results`` / ``alert_records`` /
                ``diagnosis_records``.
            older_than: if given, only purge rows whose ``deleted_at`` is
                older than this epoch timestamp.  If None, purge all.
        """
        if not self.enabled or self._conn is None:
            return 0
        if table not in ("detection_results", "alert_records", "diagnosis_records"):
            return 0
        sql = f"DELETE FROM {table} WHERE is_deleted = 1"
        params: list = []
        if older_than is not None:
            sql += " AND deleted_at IS NOT NULL AND deleted_at <= ?"
            params.append(older_than)
        try:
            with self._write_lock:
                cur = self._conn.execute(sql, params)
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("purge from %s failed", table, exc_info=True)
            return 0

    # Statuses the frontend may set via PATCH /api/alerts/{id}.
    # ``active`` is the default for measured alerts and is not meant to be
    # reassigned by hand, so it is excluded from the allow-list.
    _PATCHABLE_ALERT_STATUSES = frozenset({"pending", "confirmed", "false"})

    def update_alert_status(self, alert_id: int, status: str) -> bool:
        """Update an alert record's lifecycle status.

        Only ``pending`` / ``confirmed`` / ``false`` are accepted; ``active``
        is reserved for the initial measured-alert state. Returns True if a
        row was updated, False if the id was not found or status is invalid.
        """
        if not self.enabled or self._conn is None:
            return False
        if status not in self._PATCHABLE_ALERT_STATUSES:
            return False
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    "UPDATE alert_records SET status = ?, verified_at = ? "
                    "WHERE id = ? AND is_deleted = 0",
                    [status, time.time(), alert_id],
                )
                return cur.rowcount > 0
        except Exception:
            logger.warning("update_alert_status failed", exc_info=True)
            return False

    # Valid verdict values for llm_verdict / human_verdict columns.
    _VALID_VERDICTS = frozenset({"real", "false_alarm", "uncertain"})

    def update_alert_verdict(self, channel: str, alert_ts: float,
                             verdict: str, *, is_llm: bool = False) -> bool:
        """Set a verdict (llm or human) on an alert record.

        Locates the row by (channel, created_at).  ``is_llm=False`` writes
        ``human_verdict``, ``is_llm=True`` writes ``llm_verdict``.

        Returns True if a row was updated, False if not found or verdict
        is invalid.
        """
        if not self.enabled or self._conn is None:
            return False
        if verdict not in self._VALID_VERDICTS:
            return False
        col = "llm_verdict" if is_llm else "human_verdict"
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f"UPDATE alert_records SET {col} = ? "
                    f"WHERE channel = ? AND created_at = ? AND is_deleted = 0",
                    [verdict, channel, alert_ts],
                )
                return cur.rowcount > 0
        except Exception:
            logger.warning("update_alert_verdict failed", exc_info=True)
            return False

    # ───────────────────────────────────────────────────────────────────
    # Admin-management extensions (shared preamble for Day21 pages 3/4)
    # ───────────────────────────────────────────────────────────────────
    # Design notes:
    #   - table whitelist (SQL-injection defence; every table name comes from a hardcoded frozenset)
    #   - empty ids short-circuits to 0 (avoids an empty IN (...) error)
    #   - everything reuses _write_lock with try/except degradation; never propagates
    #   - zero behaviour change: existing delete_*/purge_deleted/update_alert_verdict are untouched
    _ADMIN_TABLES = frozenset({"detection_results", "alert_records", "diagnosis_records"})

    def _sanitize_ids(self, ids: list[int]) -> list[int]:
        """Normalise an externally-supplied id list into a deduped, positive-integer safe form."""
        seen: set[int] = set()
        out: list[int] = []
        for v in ids or []:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv > 0 and iv not in seen:
                seen.add(iv)
                out.append(iv)
        return out

    def _placeholders(self, n: int) -> str:
        """Generate a ``?,?,?`` placeholder string (sqlite3 does not accept a list expanded inline)."""
        return ",".join(["?"] * max(n, 1))

    def query_deleted(
        self,
        table: str,
        limit: int = 200,
        offset: int = 0,
        *,
        deleted_start: float | None = None,
        deleted_end: float | None = None,
        channel: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Return the **soft-deleted** rows of a table (for the recycle-bin list).

        Args:
            table: must be one of ``_ADMIN_TABLES``.
            limit: return cap, ordered by ``deleted_at DESC``.
            offset: skip the first N rows (for paging); offset=0 is backward compatible.
            deleted_start / deleted_end: filter on the soft-delete timestamp
                (``deleted_at``). Useful for "what I deleted yesterday".
            channel: sensor channel exact match (all admin tables have it).
            status: final-status filter — only effective on ``alert_records``
                (maps to status/llm_verdict/human_verdict). Ignored on other tables.

        The returned fields are aligned with ``query_alerts`` / ``query_detection``
        so the front-end can reuse the column definitions. Returns ``[]`` on failure.
        """
        if not self.enabled or self._conn is None:
            return []
        if table not in self._ADMIN_TABLES:
            return []
        try:
            limit = max(1, min(int(limit), 1000))
        except (TypeError, ValueError):
            limit = 200
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        where_sql, where_params = _build_recycle_where_clause(
            table,
            deleted_start=deleted_start, deleted_end=deleted_end,
            channel=channel, status=status,
        )
        try:
            if table == "alert_records":
                cur = self._conn.execute(
                    "SELECT id, channel, alert_type, score, message, created_at, status, "
                    "       verified_at, llm_verdict, human_verdict, raw_snapshot, deleted_at "
                    f"FROM {table} WHERE is_deleted = 1{where_sql} "
                    "ORDER BY deleted_at DESC LIMIT ? OFFSET ?",
                    where_params + [limit, offset],
                )
                rows = cur.fetchall()
                out = []
                for r in rows:
                    # raw_snapshot's last point = the telemetry value at the alert time (spec admin-section "telemetry value" column)
                    raw_snap = None
                    raw_value = None
                    if r[10]:
                        try:
                            raw_snap = json.loads(r[10])
                            if isinstance(raw_snap, list) and raw_snap:
                                last = raw_snap[-1]
                                if isinstance(last, (int, float)):
                                    raw_value = float(last)
                        except (json.JSONDecodeError, TypeError):
                            raw_snap = None
                    out.append({
                        "id": r[0], "channel": r[1], "alert_type": r[2], "score": r[3],
                        "message": r[4], "created_at": r[5], "status": r[6],
                        "verified_at": r[7], "llm_verdict": r[8], "human_verdict": r[9],
                        "raw_snapshot": raw_snap, "raw_value": raw_value,
                        "deleted_at": r[11],
                        "final_status": compute_final_status(r[6], r[8], r[9]),
                    })
                return out
            if table == "detection_results":
                cur = self._conn.execute(
                    "SELECT id, channel, timestamp, l1_score, l2_score, l3_score, "
                    "       final_score, deleted_at "
                    f"FROM {table} WHERE is_deleted = 1{where_sql} "
                    "ORDER BY deleted_at DESC LIMIT ? OFFSET ?",
                    where_params + [limit, offset],
                )
                rows = cur.fetchall()
                return [
                    {
                        "id": r[0], "channel": r[1], "timestamp": r[2],
                        "l1_score": r[3], "l2_score": r[4], "l3_score": r[5],
                        "final_score": r[6], "deleted_at": r[7],
                    }
                    for r in rows
                ]
            # diagnosis_records
            cur = self._conn.execute(
                "SELECT id, channel, alert_type, alert_ts, llm_verdict, error, created_at, deleted_at "
                f"FROM {table} WHERE is_deleted = 1{where_sql} "
                "ORDER BY deleted_at DESC LIMIT ? OFFSET ?",
                where_params + [limit, offset],
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r[0], "channel": r[1], "alert_type": r[2], "alert_ts": r[3],
                    "llm_verdict": r[4], "error": r[5], "created_at": r[6],
                    "deleted_at": r[7],
                }
                for r in rows
            ]
        except Exception:
            logger.warning("query_deleted(%s) failed", table, exc_info=True)
            return []

    def count_deleted(
        self,
        table: str,
        *,
        deleted_start: float | None = None,
        deleted_end: float | None = None,
        channel: str | None = None,
        status: str | None = None,
    ) -> int:
        """Return the total number of soft-deleted rows in a table (recycle-bin paging count).

        Args:
            table: must be one of ``_ADMIN_TABLES``.
            Filter kwargs mirror ``query_deleted`` (deleted_start / deleted_end /
            channel / status) so the page count and the rows stay consistent.

        Returns ``SELECT COUNT(*) FROM {table} WHERE is_deleted=1[AND ...]``; 0 on failure.
        """
        if not self.enabled or self._conn is None:
            return 0
        if table not in self._ADMIN_TABLES:
            return 0
        where_sql, where_params = _build_recycle_where_clause(
            table,
            deleted_start=deleted_start, deleted_end=deleted_end,
            channel=channel, status=status,
        )
        try:
            cur = self._conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE is_deleted = 1{where_sql}",
                where_params,
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            logger.warning("count_deleted(%s) failed", table, exc_info=True)
            return 0

    def delete_by_ids(self, table: str, ids: list[int]) -> int:
        """**Soft-delete** by id list (move to the recycle bin).

        ``UPDATE ... SET is_deleted=1, deleted_at=unixepoch()
           WHERE id IN (...) AND is_deleted=0``.
        Returns the number of rows actually hit (rows already in the bin are not re-counted).
        """
        if not self.enabled or self._conn is None:
            return 0
        if table not in self._ADMIN_TABLES:
            return 0
        clean = self._sanitize_ids(ids)
        if not clean:
            return 0
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f"UPDATE {table} SET is_deleted = 1, deleted_at = unixepoch() "
                    f"WHERE id IN ({self._placeholders(len(clean))}) AND is_deleted = 0",
                    clean,
                )
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("delete_by_ids(%s) failed", table, exc_info=True)
            return 0

    def restore(self, table: str, ids: list[int]) -> int:
        """**Restore** soft-deleted rows by id list (recycle-bin "restore" button).

        ``UPDATE ... SET is_deleted=0, deleted_at=NULL
           WHERE id IN (...) AND is_deleted=1``.
        Returns the number of rows actually restored.
        """
        if not self.enabled or self._conn is None:
            return 0
        if table not in self._ADMIN_TABLES:
            return 0
        clean = self._sanitize_ids(ids)
        if not clean:
            return 0
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f"UPDATE {table} SET is_deleted = 0, deleted_at = NULL "
                    f"WHERE id IN ({self._placeholders(len(clean))}) AND is_deleted = 1",
                    clean,
                )
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("restore(%s) failed", table, exc_info=True)
            return 0

    def purge_by_ids(self, table: str, ids: list[int]) -> int:
        """**Physically delete** by id list (recycle-bin "permanent delete" button).

        Unlike ``purge_deleted(table, older_than)`` — which batch-clears by a
        time window — this method deletes by an exact id list. It only affects
        rows that are **already soft-deleted**, so active data cannot be purged
        by mistake. Returns the number of rows physically deleted.
        """
        if not self.enabled or self._conn is None:
            return 0
        if table not in self._ADMIN_TABLES:
            return 0
        clean = self._sanitize_ids(ids)
        if not clean:
            return 0
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f"DELETE FROM {table} "
                    f"WHERE id IN ({self._placeholders(len(clean))}) AND is_deleted = 1",
                    clean,
                )
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("purge_by_ids(%s) failed", table, exc_info=True)
            return 0

    def update_alert_verdict_by_ids(self, ids: list[int], verdict: str,
                                    *, is_llm: bool = False) -> int:
        """Batch-write verdicts by id list (alert-management page "batch annotate").

        ``UPDATE alert_records SET <col>=? WHERE id IN (...) AND is_deleted=0``.
        Soft-deleted rows are untouched. Returns the number of rows actually updated.
        """
        if not self.enabled or self._conn is None:
            return 0
        if verdict not in self._VALID_VERDICTS:
            return 0
        clean = self._sanitize_ids(ids)
        if not clean:
            return 0
        col = "llm_verdict" if is_llm else "human_verdict"
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f"UPDATE alert_records SET {col} = ? "
                    f"WHERE id IN ({self._placeholders(len(clean))}) AND is_deleted = 0",
                    [verdict] + clean,
                )
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("update_alert_verdict_by_ids failed", exc_info=True)
            return 0

    def get_alert_by_id(self, alert_id: int) -> dict | None:
        """Fetch a single alert by id (parses raw_snapshot/score_snapshot).

        Used by the alert detail drawer. The returned fields match
        ``query_alerts_filtered``; returns None on miss.
        """
        if not self.enabled or self._conn is None:
            return None
        try:
            aid = int(alert_id)
        except (TypeError, ValueError):
            return None
        if aid <= 0:
            return None
        try:
            cur = self._conn.execute(
                "SELECT id, channel, alert_type, score, message, created_at, status, "
                "       verified_at, llm_verdict, human_verdict, raw_snapshot, "
                "       score_snapshot, ingested_at "
                "FROM alert_records WHERE id = ? AND is_deleted = 0",
                [aid],
            )
            r = cur.fetchone()
        except Exception:
            logger.warning("get_alert_by_id failed", exc_info=True)
            return None
        if not r:
            return None
        llm_v = r[8]
        human_v = r[9]
        try:
            raw_snap = json.loads(r[10]) if r[10] else None
        except (json.JSONDecodeError, TypeError):
            raw_snap = None
        try:
            score_snap = json.loads(r[11]) if r[11] else None
        except (json.JSONDecodeError, TypeError):
            score_snap = None
        return {
            "id": r[0], "channel": r[1], "alert_type": r[2], "score": r[3],
            "message": r[4], "created_at": r[5], "status": r[6],
            "verified_at": r[7], "llm_verdict": llm_v, "human_verdict": human_v,
            "raw_snapshot": raw_snap, "score_snapshot": score_snap,
            "ingested_at": r[12],
            "final_status": compute_final_status(r[6], llm_v, human_v),
        }

    def count_alerts_filtered(
        self,
        *,
        channel: str | None = None,
        alert_type: str | None = None,
        status: str | None = None,
        verdict: str | None = None,
        llm_verdict: str | None = None,
        human_verdict: str | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> int:
        """Count the alert_records rows matching the filters (for paging total-page calc).

        The filter params are identical to ``query_alerts_filtered`` (without limit/offset).
        Returns 0 on failure.
        """
        if not self.enabled or self._conn is None:
            return 0
        where_sql, params = _build_alert_where_clause(
            channel=channel, alert_type=alert_type, status=status,
            verdict=verdict, llm_verdict=llm_verdict, human_verdict=human_verdict,
            start_ts=start_ts, end_ts=end_ts,
        )
        sql = "SELECT COUNT(*) FROM alert_records WHERE is_deleted = 0" + where_sql
        try:
            cur = self._conn.execute(sql, params)
            return int(cur.fetchone()[0])
        except Exception:
            logger.warning("count_alerts_filtered failed", exc_info=True)
            return 0

    def query_alerts_filtered(
        self,
        *,
        channel: str | None = None,
        alert_type: str | None = None,
        status: str | None = None,
        verdict: str | None = None,
        llm_verdict: str | None = None,
        human_verdict: str | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Filtered query over alert_records (for the alert-management page list).

        All params optional; None means "no filter".
          - ``verdict``: combined (matches if llm_verdict or human_verdict equals it)
          - ``llm_verdict``: filter on the LLM diagnosis alone; 'none' means undiagnosed (IS NULL)
          - ``human_verdict``: same as above
          - ``offset``: paging offset (default 0, backward compatible). Combined with ``limit`` for paging.
        The returned fields match ``query_alerts`` plus ``ingested_at``.
        Returns ``[]`` on failure.
        """
        if not self.enabled or self._conn is None:
            return []
        try:
            limit = max(1, min(int(limit), 1000))
        except (TypeError, ValueError):
            limit = 50
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        where_sql, params = _build_alert_where_clause(
            channel=channel, alert_type=alert_type, status=status,
            verdict=verdict, llm_verdict=llm_verdict, human_verdict=human_verdict,
            start_ts=start_ts, end_ts=end_ts,
        )
        sql = (
            "SELECT id, channel, alert_type, score, message, created_at, status, "
            "       verified_at, llm_verdict, human_verdict, raw_snapshot, "
            "       score_snapshot, ingested_at "
            "FROM alert_records WHERE is_deleted = 0" + where_sql
            + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.append(limit)
        params.append(offset)
        try:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        except Exception:
            logger.warning("query_alerts_filtered failed", exc_info=True)
            return []
        results = []
        for r in reversed(rows):  # chronological
            llm_v = r[8]
            human_v = r[9]
            try:
                raw_snap = json.loads(r[10]) if r[10] else None
            except (json.JSONDecodeError, TypeError):
                raw_snap = None
            try:
                score_snap = json.loads(r[11]) if r[11] else None
            except (json.JSONDecodeError, TypeError):
                score_snap = None
            results.append({
                "id": r[0], "channel": r[1], "alert_type": r[2], "score": r[3],
                "message": r[4], "created_at": r[5], "status": r[6],
                "verified_at": r[7], "llm_verdict": llm_v, "human_verdict": human_v,
                "raw_snapshot": raw_snap, "score_snapshot": score_snap,
                "ingested_at": r[12],
                "final_status": compute_final_status(r[6], llm_v, human_v),
            })
        return results

    def insert_alert_manual(
        self,
        channel: str,
        score: float,
        message: str = "",
        *,
        created_at: float | None = None,
        raw_snapshot: list | None = None,
        score_snapshot: list | None = None,
        status: str = "active",
    ) -> int | None:
        """Manually insert an alert record (alert-management page "create" button).

        Defaults to ``alert_type='measured'`` and ``status='active'`` (can be PATCHed on the page afterwards).
        Synchronous insert + immediate commit (not queued, so the returned id is usable).
        Returns the new row id; None on failure.
        """
        if not self.enabled or self._conn is None:
            return None
        if not channel or not isinstance(channel, str):
            return None
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None
        ts = float(created_at) if created_at is not None else time.time()
        raw_json = json.dumps(raw_snapshot) if raw_snapshot is not None else None
        score_json = json.dumps(score_snapshot) if score_snapshot is not None else None
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    "INSERT INTO alert_records "
                    "(channel, alert_type, score, message, created_at, status, "
                    " raw_snapshot, score_snapshot, ingested_at, is_deleted) "
                    "VALUES (?, 'measured', ?, ?, ?, ?, ?, ?, unixepoch(), 0)",
                    [channel, score_f, message or "", ts, status, raw_json, score_json],
                )
                self._conn.commit()
                return cur.lastrowid if cur.lastrowid is not None else None
        except Exception:
            logger.warning("insert_alert_manual failed", exc_info=True)
            return None

    # ───────────────────────────────────────────────────────────────────
    # Per-channel telemetry: soft-delete / restore / purge / recycle-bin
    # (v1.2 data-management page + recycle bin)
    # ───────────────────────────────────────────────────────────────────
    #
    # These mirror the alert/detection recycle-bin API (delete_by_ids /
    # restore / purge_by_ids) but operate on a specific telemetry_<channel>
    # table by rowid instead of a global ``id`` column. rowid is per-table,
    # so every method takes ``(channel, rowids)`` — the view layer is
    # responsible for validating the channel name.
    #
    # Unlike ``delete_history`` (HARD delete, kept for the DB panel), these
    # methods implement the soft-delete recycle-bin semantics the admin UI
    # needs: soft_delete_tel moves rows to the bin (deleted_at = now),
    # restore_tel brings them back, purge_tel physically removes rows that
    # are already in the bin.

    def soft_delete_tel(self, channel: str, rowids: list[int]) -> int:
        """Soft-delete telemetry rows by rowid (move to the recycle bin).

        ``UPDATE telemetry_<ch> SET deleted_at = unixepoch()
           WHERE rowid IN (...) AND deleted_at IS NULL``.
        Returns the number of rows actually moved (rows already in the bin
        are not re-counted).
        """
        if not self.enabled or self._conn is None:
            return 0
        if not channel or not isinstance(channel, str):
            return 0
        clean = self._sanitize_ids(rowids)
        if not clean:
            return 0
        table = self._ensure_tel_table(channel)
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f'UPDATE "{table}" SET deleted_at = unixepoch() '
                    f'WHERE rowid IN ({self._placeholders(len(clean))}) '
                    f'AND deleted_at IS NULL',
                    clean,
                )
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("soft_delete_tel(%s) failed", channel, exc_info=True)
            return 0

    def restore_tel(self, channel: str, rowids: list[int]) -> int:
        """Restore soft-deleted telemetry rows by rowid (recycle-bin restore).

        ``UPDATE telemetry_<ch> SET deleted_at = NULL
           WHERE rowid IN (...) AND deleted_at IS NOT NULL``.
        Returns the number of rows actually restored.
        """
        if not self.enabled or self._conn is None:
            return 0
        if not channel or not isinstance(channel, str):
            return 0
        clean = self._sanitize_ids(rowids)
        if not clean:
            return 0
        table = self._ensure_tel_table(channel)
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f'UPDATE "{table}" SET deleted_at = NULL '
                    f'WHERE rowid IN ({self._placeholders(len(clean))}) '
                    f'AND deleted_at IS NOT NULL',
                    clean,
                )
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("restore_tel(%s) failed", channel, exc_info=True)
            return 0

    def purge_tel(self, channel: str, rowids: list[int]) -> int:
        """Physically delete telemetry rows by rowid (recycle-bin permanent delete).

        Only affects rows that are **already soft-deleted**, so live data
        cannot be purged by mistake. Irreversible. Returns the number of
        rows physically removed.
        """
        if not self.enabled or self._conn is None:
            return 0
        if not channel or not isinstance(channel, str):
            return 0
        clean = self._sanitize_ids(rowids)
        if not clean:
            return 0
        table = self._ensure_tel_table(channel)
        try:
            with self._write_lock:
                cur = self._conn.execute(
                    f'DELETE FROM "{table}" '
                    f'WHERE rowid IN ({self._placeholders(len(clean))}) '
                    f'AND deleted_at IS NOT NULL',
                    clean,
                )
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            logger.warning("purge_tel(%s) failed", channel, exc_info=True)
            return 0

    def query_tel_deleted(
        self,
        channel: str,
        *,
        limit: int = 200,
        offset: int = 0,
        deleted_start: float | None = None,
        deleted_end: float | None = None,
    ) -> list[dict]:
        """List soft-deleted telemetry rows for a channel (recycle-bin view).

        Returns rows newest-deleted-first (ORDER BY deleted_at DESC) with
        the same fields as ``query_tel_page`` so the front end can reuse
        the column definitions. Returns ``[]`` on failure or empty channel.

        ``deleted_start`` / ``deleted_end`` optionally filter on the soft-delete
        timestamp (``deleted_at``); both bounds are inclusive.
        """
        if not self.enabled or self._conn is None:
            return []
        if not channel or not isinstance(channel, str):
            return []
        try:
            limit = max(1, min(int(limit), 1000))
        except (TypeError, ValueError):
            limit = 200
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        where_parts: list[str] = ["deleted_at IS NOT NULL"]
        where_params: list = []
        if deleted_start is not None:
            where_parts.append("deleted_at >= ?")
            where_params.append(deleted_start)
        if deleted_end is not None:
            where_parts.append("deleted_at <= ?")
            where_params.append(deleted_end)
        where_sql = " AND ".join(where_parts)
        table = self._ensure_tel_table(channel)
        try:
            cur = self._conn.execute(
                f'SELECT rowid AS id, timestamp, raw_value, anomaly_score, '
                f'predicted_value, predicted_anomaly_score, origin_ts, '
                f'ingested_at, deleted_at, origin '
                f'FROM "{table}" WHERE {where_sql} '
                f'ORDER BY deleted_at DESC LIMIT ? OFFSET ?',
                where_params + [limit, offset],
            )
            rows = cur.fetchall()
        except Exception:
            logger.warning("query_tel_deleted(%s) failed", channel, exc_info=True)
            return []
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "raw_value": r[2],
                "anomaly_score": r[3],
                "predicted_value": r[4],
                "predicted_anomaly_score": r[5],
                "origin_ts": r[6],
                "ingested_at": r[7],
                "deleted_at": r[8],
                "origin": r[9],
            }
            for r in rows
        ]

    def count_tel_deleted(
        self,
        channel: str,
        *,
        deleted_start: float | None = None,
        deleted_end: float | None = None,
    ) -> int:
        """Count soft-deleted telemetry rows for a channel (recycle-bin paging).

        Filter kwargs mirror ``query_tel_deleted`` so the page count and the
        rows stay in sync. Returns 0 on failure or empty channel.
        """
        if not self.enabled or self._conn is None:
            return 0
        if not channel or not isinstance(channel, str):
            return 0
        where_parts: list[str] = ["deleted_at IS NOT NULL"]
        where_params: list = []
        if deleted_start is not None:
            where_parts.append("deleted_at >= ?")
            where_params.append(deleted_start)
        if deleted_end is not None:
            where_parts.append("deleted_at <= ?")
            where_params.append(deleted_end)
        where_sql = " AND ".join(where_parts)
        table = self._ensure_tel_table(channel)
        try:
            cur = self._conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE {where_sql}',
                where_params,
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            logger.warning("count_tel_deleted(%s) failed", channel, exc_info=True)
            return 0

    # ───────────────────────────────────────────────────────────────────
    # Per-channel telemetry: manual insert (origin='manual')
    # (v1.2 data-management page "add point" action)
    # ───────────────────────────────────────────────────────────────────

    def insert_tel_manual(
        self,
        channel: str,
        timestamp: float,
        raw_value: float | None = None,
        anomaly_score: float | None = None,
        *,
        predicted_value: float | None = None,
    ) -> bool:
        """Manually insert / upsert a telemetry row tagged ``origin='manual'``.

        Uses ``INSERT OR REPLACE`` keyed by timestamp: if a row already
        exists at that timestamp (raw or predicted), the new manual values
        overwrite it. The row is tagged ``origin='manual'`` so the front end
        can colour it separately from acquisition points.

        This method does **not** touch acquisition bookkeeping
        (``t_acq_start`` anchoring / block sequence numbers): manual rows
        are excluded from the acquisition pipeline, they only land in the
        per-channel telemetry table for display / export.

        Returns True on success, False on failure.
        """
        if not self.enabled or self._conn is None:
            return False
        if not channel or not isinstance(channel, str):
            return False
        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            return False
        try:
            raw_f = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            raw_f = None
        try:
            score_f = float(anomaly_score) if anomaly_score is not None else None
        except (TypeError, ValueError):
            score_f = None
        try:
            pred_f = float(predicted_value) if predicted_value is not None else None
        except (TypeError, ValueError):
            pred_f = None
        table = self._ensure_tel_table(channel)
        try:
            with self._write_lock:
                self._conn.execute(
                    f'INSERT OR REPLACE INTO "{table}" '
                    f'(timestamp, raw_value, anomaly_score, predicted_value, '
                    f' ingested_at, origin) '
                    f'VALUES (?, ?, ?, ?, unixepoch(), \'manual\')',
                    [ts, raw_f, score_f, pred_f],
                )
                self._conn.commit()
            return True
        except Exception:
            logger.warning("insert_tel_manual(%s) failed", channel, exc_info=True)
            return False

    # ───────────────────────────────────────────────────────────────────
    # Per-channel telemetry: streaming export generator
    # (v1.2 data-management page "export CSV")
    # ───────────────────────────────────────────────────────────────────

    def iter_tel_rows(
        self,
        channel: str,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        batch_size: int = 1000,
    ) -> Iterator[dict]:
        """Stream telemetry rows for memory-bounded CSV export.

        Yields one row dict at a time (same shape as ``query_tel_page``
        minus the soft-delete flag — only live rows are exported). Uses
        keyset pagination on ``timestamp`` (the PRIMARY KEY) so memory use
        stays bounded by ``batch_size`` regardless of total row count.

        Args:
            channel:   per-channel table to stream.
            start_ts:  optional inclusive lower timestamp bound.
            end_ts:    optional inclusive upper timestamp bound.
            batch_size: rows fetched per internal query (default 1000).
        """
        if not self.enabled or self._conn is None:
            return
        if not channel or not isinstance(channel, str):
            return
        try:
            batch_size = max(1, int(batch_size))
        except (TypeError, ValueError):
            batch_size = 1000
        table = self._ensure_tel_table(channel)
        # Export chronological (ascending) for a natural CSV reading order.
        # Keyset pagination walks timestamp upward. The first batch uses an
        # inclusive ``>= start_ts``; subsequent batches use ``> last_ts`` so
        # the boundary row is never re-emitted. timestamp is the PRIMARY KEY
        # (unique), so a strict greater-than is safe and never skips rows.
        first_batch = True
        cursor_ts: float = start_ts if start_ts is not None else float("-inf")
        try:
            while True:
                if first_batch:
                    where_parts = ["timestamp >= ?"]
                else:
                    where_parts = ["timestamp > ?"]
                params: list = [cursor_ts]
                if end_ts is not None:
                    where_parts.append("timestamp <= ?")
                    params.append(end_ts)
                sql = (
                    f'SELECT rowid AS id, timestamp, raw_value, anomaly_score, '
                    f'predicted_value, predicted_anomaly_score, origin_ts, '
                    f'ingested_at, deleted_at, origin '
                    f'FROM "{table}" WHERE {" AND ".join(where_parts)} '
                    f'AND deleted_at IS NULL '
                    f'ORDER BY timestamp ASC LIMIT ?'
                )
                params.append(batch_size)
                cur = self._conn.execute(sql, params)
                rows = cur.fetchall()
                if not rows:
                    return
                last_ts = rows[-1][1]
                for r in rows:
                    yield {
                        "id": r[0],
                        "timestamp": r[1],
                        "raw_value": r[2],
                        "anomaly_score": r[3],
                        "predicted_value": r[4],
                        "predicted_anomaly_score": r[5],
                        "origin_ts": r[6],
                        "ingested_at": r[7],
                        "deleted_at": r[8],
                        "origin": r[9],
                    }
                # Fewer rows than batch_size → this was the last page.
                if len(rows) < batch_size:
                    return
                # No forward progress (shouldn't happen since timestamp is a
                # unique PK) — break to avoid an infinite loop.
                if last_ts <= cursor_ts and not first_batch:
                    return
                cursor_ts = last_ts
                first_batch = False
        except Exception:
            logger.warning("iter_tel_rows(%s) failed", channel, exc_info=True)
            return

    def stats(self) -> dict:
        """Return row counts for each table (for monitoring)."""
        if not self.enabled or self._conn is None:
            return {"enabled": False}
        try:
            # Sum row counts across all per-channel telemetry_* tables
            tables = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'telemetry_%'"
            ).fetchall()
            n_tel = 0
            tel_by_channel: dict[str, int] = {}
            for (t,) in tables:
                cnt = self._conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                n_tel += cnt
                ch_name = t[len("telemetry_"):]
                tel_by_channel[ch_name] = cnt
            n_det = self._conn.execute(
                "SELECT COUNT(*) FROM detection_results WHERE is_deleted = 0"
            ).fetchone()[0]
            n_alert = self._conn.execute(
                "SELECT COUNT(*) FROM alert_records WHERE is_deleted = 0"
            ).fetchone()[0]
        except Exception:
            return {"enabled": True, "error": "query_failed"}
        return {
            "enabled": True,
            "db_path": str(self.db_path),
            "telemetry": n_tel,
            "telemetry_by_channel": tel_by_channel,
            "detection_results": n_det,
            "alert_records": n_alert,
            "queue_pending": self._queue.qsize(),
        }
