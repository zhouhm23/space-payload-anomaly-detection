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
-- Raw telemetry samples (one row per sample point)
CREATE TABLE IF NOT EXISTS raw_telemetry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel     TEXT    NOT NULL,
    raw         REAL,
    score       REAL,
    received_at REAL    NOT NULL,
    ingested_at REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_raw_channel_time ON raw_telemetry(channel, received_at);

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

-- Predicted values + predicted anomaly scores (batch-per-block)
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel         TEXT    NOT NULL,
    origin_ts       REAL    NOT NULL,   -- timestamp of the last measured point
    predict_start   REAL    NOT NULL,   -- first predicted-point timestamp
    predict_end     REAL    NOT NULL,   -- last predicted-point timestamp
    prediction      TEXT    NOT NULL,   -- JSON array of predicted raw values
    predict_scores  TEXT,               -- JSON array of predicted anomaly scores
    model           TEXT,               -- 'ttm-r3' / 'linear'
    ingested_at     REAL    NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_pred_channel_origin ON predictions(channel, origin_ts);
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
    ) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.enabled = enabled

        self._queue: queue.Queue[list | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()  # guards _conn writes

        if self.enabled:
            self._init_db()

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
        """
        if not self.enabled:
            return
        batch: list = []
        for e in entries:
            batch.append([
                "raw", e["channel"], float(e["raw"]),
                float(e["score"]) if e.get("score") is not None else None,
                float(e["received_at"]),
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
            (r[1], r[2], r[3], r[4])
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
        pred_rows = [
            (r[1], r[2], r[3], r[4], r[5], r[6], r[7])
            for r in flat if r[0] == "pred"
        ]

        try:
            with self._write_lock:
                if raw_rows:
                    self._conn.executemany(
                        "INSERT INTO raw_telemetry(channel, raw, score, received_at) VALUES (?,?,?,?)",
                        raw_rows,
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
                if pred_rows:
                    self._conn.executemany(
                        """INSERT INTO predictions
                           (channel, origin_ts, predict_start, predict_end,
                            prediction, predict_scores, model)
                           VALUES (?,?,?,?,?,?,?)""",
                        pred_rows,
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
    ) -> None:
        if not self.enabled:
            return
        self._queue.put([
            "pred",
            channel,
            float(origin_ts),
            float(predict_start),
            float(predict_end),
            json.dumps(prediction, ensure_ascii=False),
            json.dumps(predict_scores, ensure_ascii=False) if predict_scores is not None else None,
            model,
        ])

    def query_predictions(
        self,
        channel: str,
        limit: int = 50,
    ) -> list[dict]:
        """Query recent prediction batches for a channel."""
        if not self.enabled or self._conn is None:
            return []
        sql = (
            "SELECT channel, origin_ts, predict_start, predict_end, "
            "prediction, predict_scores, model "
            "FROM predictions WHERE channel = ? "
            "ORDER BY origin_ts DESC LIMIT ?"
        )
        try:
            cur = self._conn.execute(sql, [channel, limit])
            rows = cur.fetchall()
        except Exception:
            logger.warning("query_predictions failed", exc_info=True)
            return []
        results = []
        for r in reversed(rows):
            try:
                pred = json.loads(r[4]) if r[4] else []
            except Exception:
                pred = []
            try:
                pscores = json.loads(r[5]) if r[5] else []
            except Exception:
                pscores = []
            results.append({
                "channel": r[0],
                "origin_ts": r[1],
                "predict_start": r[2],
                "predict_end": r[3],
                "prediction": pred,
                "predict_scores": pscores,
                "model": r[6],
            })
        return results

    # ------------------------------------------------------------------
    # Window query — latest N raw + predictions within that window
    # ------------------------------------------------------------------

    def query_window(
        self,
        channel: str,
        count: int = 512,
        end_ts: float | None = None,
    ) -> dict:
        """Return the latest ``count`` raw telemetry points for *channel*
        plus any prediction batches whose origin falls within the window.

        Args:
            channel:  channel name.
            count:    window length in points (100–10000).
            end_ts:   right-edge epoch seconds.  If None, use the latest
                      row in the DB (auto-scroll).

        Returns dict::

            {"channel": ..., "count": N, "end_ts": ..., "start_ts": ...,
             "raw": [{raw, score, received_at}, ...],
             "predictions": [{origin_ts, predict_start, predict_end,
                              prediction, predict_scores}, ...]}
        """
        if not self.enabled or self._conn is None:
            return {"channel": channel, "count": 0, "raw": [], "predictions": []}

        try:
            if end_ts is None:
                row = self._conn.execute(
                    "SELECT MAX(received_at) FROM raw_telemetry WHERE channel = ?",
                    [channel],
                ).fetchone()
                end_ts = row[0] if row and row[0] is not None else None

            if end_ts is None:
                return {"channel": channel, "count": 0, "end_ts": None,
                        "start_ts": None, "raw": [], "predictions": []}

            # Fetch the *count* rows whose received_at <= end_ts, ordered
            # descending then reversed so the result is chronological.
            cur = self._conn.execute(
                "SELECT raw, score, received_at FROM raw_telemetry "
                "WHERE channel = ? AND received_at <= ? "
                "ORDER BY received_at DESC LIMIT ?",
                [channel, end_ts, count],
            )
            rows = list(reversed(cur.fetchall()))

            # Deduplicate by received_at (auto-poll can produce overlapping
            # blocks) and ensure strictly ascending timestamps so the chart
            # renders a clean monotonic line.
            # Use 1ms granularity (round to 3 decimal places) — finer
            # precision lets near-duplicate timestamps through and causes
            # visual zig-zag when two data streams interleave.
            seen: set = set()
            deduped: list = []
            for r in rows:
                ts = round(r[2], 3)  # 1ms precision for dedup
                if ts in seen:
                    continue
                seen.add(ts)
                deduped.append(r)

            start_ts = deduped[0][2] if deduped else end_ts

            raw = [
                {"raw": r[0], "score": r[1], "received_at": r[2]} for r in deduped
            ]

            # Predictions: only return the single most recent batch within
            # the window (whose origin is closest to the right edge).
            # Returning multiple overlapping batches causes the frontend to
            # draw interlocking dashed segments that look garbled.
            pcur = self._conn.execute(
                "SELECT origin_ts, predict_start, predict_end, "
                "prediction, predict_scores, model "
                "FROM predictions WHERE channel = ? "
                "AND origin_ts >= ? AND origin_ts <= ? "
                "ORDER BY origin_ts DESC LIMIT 1",
                [channel, start_ts, end_ts],
            )
            preds = []
            prow = pcur.fetchone()
            if prow:
                try:
                    pred = json.loads(prow[3]) if prow[3] else []
                except Exception:
                    pred = []
                try:
                    pscores = json.loads(prow[4]) if prow[4] else []
                except Exception:
                    pscores = []
                preds.append({
                    "origin_ts": prow[0], "predict_start": prow[1],
                    "predict_end": prow[2], "prediction": pred,
                    "predict_scores": pscores, "model": prow[5],
                })

            return {
                "channel": channel,
                "count": len(raw),
                "end_ts": end_ts,
                "start_ts": start_ts,
                "raw": raw,
                "predictions": preds,
            }
        except Exception:
            logger.warning("query_window failed", exc_info=True)
            return {"channel": channel, "count": 0, "raw": [], "predictions": []}

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
        """Query raw telemetry history."""
        if not self.enabled or self._conn is None:
            return []
        sql = "SELECT channel, raw, score, received_at FROM raw_telemetry WHERE 1=1"
        params: list = []
        if channel:
            sql += " AND channel = ?"
            params.append(channel)
        if start_time is not None:
            sql += " AND received_at >= ?"
            params.append(start_time)
        if end_time is not None:
            sql += " AND received_at <= ?"
            params.append(end_time)
        sql += " ORDER BY received_at DESC LIMIT ?"
        params.append(limit)
        try:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        except Exception:
            logger.warning("query_history failed", exc_info=True)
            return []
        return [
            {"channel": r[0], "raw": r[1], "score": r[2], "received_at": r[3]}
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
        """Query persisted alert records."""
        if not self.enabled or self._conn is None:
            return []
        sql = (
            "SELECT channel, alert_type, score, message, created_at, status, verified_at "
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
                "channel": r[0],
                "alert_type": r[1],
                "score": r[2],
                "message": r[3],
                "created_at": r[4],
                "status": r[5],
                "verified_at": r[6],
            }
            for r in reversed(rows)
        ]

    def stats(self) -> dict:
        """Return row counts for each table (for monitoring)."""
        if not self.enabled or self._conn is None:
            return {"enabled": False}
        try:
            n_raw = self._conn.execute("SELECT COUNT(*) FROM raw_telemetry").fetchone()[0]
            n_det = self._conn.execute("SELECT COUNT(*) FROM detection_results").fetchone()[0]
            n_alert = self._conn.execute("SELECT COUNT(*) FROM alert_records").fetchone()[0]
            n_pred = self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        except Exception:
            return {"enabled": True, "error": "query_failed"}
        return {
            "enabled": True,
            "db_path": str(self.db_path),
            "raw_telemetry": n_raw,
            "detection_results": n_det,
            "alert_records": n_alert,
            "predictions": n_pred,
            "queue_pending": self._queue.qsize(),
        }
