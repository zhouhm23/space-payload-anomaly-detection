"""Agent-friendly CLI to export telemetry data.

Mirrors ``GET /api/export`` but adds two CLI-friendly conveniences the HTTP
endpoint lacks:

  * ``--last 1h|1d|30m`` — relative time window (agents often want "the
    last hour of data" without computing epoch timestamps).
  * ``-o file`` — write directly to a file (default: stdout).

Usage::

    # Absolute time window
    manage.py export telemetry --channels C-1,VS-sine \\
        --start 1752633600 --end 1752633700

    # Relative time window (CLI convenience)
    manage.py export telemetry --channels C-1 --last 1h
    manage.py export telemetry --channels C-1 --last 30m

    # JSON instead of CSV (for agent pipelines)
    manage.py export telemetry --channels C-1 --last 1h --format json

    # Write to file
    manage.py export telemetry --channels C-1 --last 1d -o data.csv

The ``--format`` flag here controls the *data* format (csv vs json), unlike
other commands where it switches between text table and JSON envelope.
This matches user intuition: "export as csv" vs "export as json".
"""

from __future__ import annotations

import csv
import io
import json as json_mod
import re
import time
from datetime import datetime, timezone

from django.core.management.base import BaseCommand

from phm_site.services_bridge import get_container

from ._common import error_payload, ok_payload

# Max rows per channel — matches the HTTP /api/export cap so CLI and API
# return the same data volume for identical queries.
_PER_CHANNEL_LIMIT = 100_000

# Parse "--last 1h" / "30m" / "2d" → seconds.
_LAST_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([hmd])\s*$", re.IGNORECASE)
_LAST_UNITS = {"h": 3600, "m": 60, "d": 86400}


def _parse_last(spec: str) -> float | None:
    """Return the seconds equivalent of '--last 1h', or None if unparseable."""
    m = _LAST_RE.match(spec)
    if not m:
        return None
    return float(m.group(1)) * _LAST_UNITS[m.group(2).lower()]


class Command(BaseCommand):
    help = "Export telemetry data as CSV or JSON."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "kind", choices=["telemetry"],
            help="Export kind (currently only 'telemetry').",
        )
        parser.add_argument(
            "--channels", required=True,
            help="Comma-separated channel names (e.g. C-1,VS-sine).",
        )
        parser.add_argument(
            "--start", type=float,
            help="Start timestamp (epoch seconds). Required unless --last is given.",
        )
        parser.add_argument(
            "--end", type=float,
            help="End timestamp (epoch seconds). Defaults to now.",
        )
        parser.add_argument(
            "--last",
            help="Relative window shortcut: '1h', '30m', '2d'. Overrides --start/--end.",
        )
        parser.add_argument(
            "--limit", type=int, default=_PER_CHANNEL_LIMIT,
            help=f"Max rows per channel (default {_PER_CHANNEL_LIMIT}).",
        )
        parser.add_argument(
            "-o", "--output", default="",
            help="Write to this file (default: stdout).",
        )
        parser.add_argument(
            "--format", choices=["csv", "json"], default="csv",
            help="Data format (default csv; json for agent pipelines).",
        )

    def handle(self, *args, **options) -> None:
        kind = options["kind"]
        if kind != "telemetry":
            self.stderr.write(self.style.ERROR(f"Unknown export kind: {kind}"))
            return
        channels = [c.strip() for c in options["channels"].split(",") if c.strip()]
        if not channels:
            self.stderr.write(self.style.ERROR("--channels is empty after parsing."))
            return

        # Resolve the time window.  --last wins over --start/--end.
        start, end = self._resolve_window(options)
        if start is None:
            self.stderr.write(self.style.ERROR(
                "Need --start (and optionally --end) or --last (e.g. --last 1h)."
            ))
            return

        c = get_container()
        all_rows: list[dict] = []
        for ch in channels:
            rows = c.sqlite.query_history(
                channel=ch, start_time=start, end_time=end,
                limit=options["limit"],
            )
            all_rows.extend(rows)
        all_rows.sort(key=lambda r: r.get("received_at", 0))

        fmt = options["format"]
        out_path = options["output"]
        if fmt == "json":
            content = self._to_json(all_rows)
        else:
            content = self._to_csv(all_rows)

        if out_path:
            # UTF-8 with BOM — matches HTTP /api/export so Excel opens it cleanly.
            with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
                f.write(content)
            payload = ok_payload(
                rows=len(all_rows), channels=channels,
                start=start, end=end, format=fmt, written_to=out_path,
            )
            self.stdout.write(self.style.SUCCESS(
                f"Wrote {payload['rows']} rows to {out_path}"
            ))
        else:
            # Stream to stdout.  No JSON envelope here — the data itself is
            # already csv or json; wrapping it would break `> file.csv`.
            self.stdout.write(content)

    def _resolve_window(self, options) -> tuple[float | None, float | None]:
        """Return (start, end) epoch seconds, or (None, None) if unresolvable."""
        if options["last"]:
            secs = _parse_last(options["last"])
            if secs is None:
                self.stderr.write(self.style.ERROR(
                    f"Invalid --last value: {options['last']!r}. "
                    "Expected like '1h', '30m', '2d'."
                ))
                return None, None
            end = options["end"] if options["end"] is not None else time.time()
            return end - secs, end
        start = options["start"]
        end = options["end"] if options["end"] is not None else time.time()
        return start, end

    def _to_csv(self, rows: list[dict]) -> str:
        """Render rows to CSV (column order matches HTTP /api/export)."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["channel", "timestamp", "raw_value", "anomaly_score", "received_at_iso"])
        for r in rows:
            ts = r.get("received_at")
            iso = (datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                   if ts is not None else "")
            writer.writerow([
                r.get("channel", ""),
                ts if ts is not None else "",
                r.get("raw", ""),
                r.get("score", ""),
                iso,
            ])
        return buf.getvalue()

    def _to_json(self, rows: list[dict]) -> str:
        """Render rows to a JSON array (one object per telemetry point)."""
        out = []
        for r in rows:
            ts = r.get("received_at")
            iso = (datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                   if ts is not None else None)
            out.append({
                "channel": r.get("channel", ""),
                "timestamp": ts,
                "raw_value": r.get("raw"),
                "anomaly_score": r.get("score"),
                "received_at_iso": iso,
            })
        return json_mod.dumps(out, ensure_ascii=False, indent=2)
