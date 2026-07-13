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
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SQLiteStore"]

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "phm.db"

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
    ingested_at     REAL    NOT NULL DEFAULT (unixepoch())
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
    ingested_at REAL    NOT NULL DEFAULT (unixepoch())
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
    created_at      REAL    NOT NULL DEFAULT (unixepoch())
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
        self._conn.executescript(_SCHEMA)
        logger.info("SQLiteStore initialised: %s", self.db_path)

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
                ingested_at                 REAL    NOT NULL DEFAULT (unixepoch())
            )
        """)
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table}_ts" ON "{table}"(timestamp)'
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
        ``message``, ``time`` (or ``created_at``), ``status``.
        """
        if not self.enabled:
            return
        alert_type = alert.get("alert_type") or alert.get("type", "measured")
        created = alert.get("created_at") or alert.get("time", time.time())
        self._queue.put([
            "alert",
            alert.get("channel", ""),
            alert_type,
            alert.get("score"),
            alert.get("message", ""),
            float(created),
            alert.get("status", "active"),
            alert.get("verified_at"),
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
            (r[1], r[2], r[3], r[4], r[5], r[6], r[7])
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
                            l2_score, l3_score, l3_rules, final_score)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        det_rows,
                    )
                if alert_rows:
                    self._conn.executemany(
                        """INSERT INTO alert_records
                           (channel, alert_type, score, message, created_at, status, verified_at)
                           VALUES (?,?,?,?,?,?,?)""",
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
            "FROM detection_results WHERE 1=1"
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
            "SELECT id, channel, alert_type, score, message, created_at, status, verified_at "
            "FROM alert_records ORDER BY created_at DESC LIMIT ?"
        )
        try:
            cur = self._conn.execute(sql, [limit])
            rows = cur.fetchall()
        except Exception:
            logger.warning("query_alerts failed", exc_info=True)
            return []
        return [
            {
                "id": r[0],
                "channel": r[1],
                "alert_type": r[2],
                "score": r[3],
                "message": r[4],
                "created_at": r[5],
                "status": r[6],
                "verified_at": r[7],
            }
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # Diagnosis cache (synchronous, for POST /api/diagnosis)
    # ------------------------------------------------------------------

    def get_diagnosis(self, channel: str, alert_type: str, alert_ts: float) -> dict | None:
        """Return cached diagnosis for (channel, alert_type, alert_ts) or None."""
        if not self.enabled or self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT diagnosis, context_summary, elapsed_sec, error, created_at "
                "FROM diagnosis_records WHERE channel=? AND alert_type=? AND alert_ts=?",
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
            }
        except Exception:
            logger.warning("get_diagnosis failed", exc_info=True)
            return None

    def save_diagnosis(self, channel: str, alert_type: str, alert_ts: float,
                       diagnosis: str, context_summary: dict,
                       elapsed_sec: float, error: str | None) -> None:
        """Insert or replace a diagnosis record (keyed by channel+type+alert_ts)."""
        if not self.enabled or self._conn is None:
            return
        import json as _json
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO diagnosis_records "
                "(channel, alert_type, alert_ts, diagnosis, context_summary, "
                " elapsed_sec, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, unixepoch())",
                [channel, alert_type, alert_ts, diagnosis,
                 _json.dumps(context_summary, ensure_ascii=False),
                 elapsed_sec, error],
            )
            self._conn.commit()
        except Exception:
            logger.warning("save_diagnosis failed", exc_info=True)

    def list_diagnosis_keys(self, limit: int = 200) -> list[dict]:
        """Return cached diagnosis keys (channel, alert_type, alert_ts)."""
        if not self.enabled or self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                "SELECT channel, alert_type, alert_ts FROM diagnosis_records "
                "ORDER BY created_at DESC LIMIT ?",
                [limit],
            )
            return [
                {"channel": r[0], "alert_type": r[1], "alert_ts": r[2]}
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
        """Delete detection-result rows matching the filter.

        Note the time column is ``timestamp`` here, not ``received_at``.
        """
        return self._delete_rows(
            "detection_results", "timestamp",
            channel=channel, start_time=start_time, end_time=end_time,
        )

    def _delete_rows(
        self,
        table: str,
        time_col: str,
        *,
        channel: str | None,
        start_time: float | None,
        end_time: float | None,
    ) -> int:
        """Shared DELETE builder for a table with ``channel`` + time column."""
        if not self.enabled or self._conn is None:
            return 0
        sql = f"DELETE FROM {table} WHERE 1=1"
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
            logger.warning("delete from %s failed", table, exc_info=True)
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
                    "UPDATE alert_records SET status = ?, verified_at = ? WHERE id = ?",
                    [status, time.time(), alert_id],
                )
                return cur.rowcount > 0
        except Exception:
            logger.warning("update_alert_status failed", exc_info=True)
            return False

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
            n_det = self._conn.execute("SELECT COUNT(*) FROM detection_results").fetchone()[0]
            n_alert = self._conn.execute("SELECT COUNT(*) FROM alert_records").fetchone()[0]
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
