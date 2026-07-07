"""GET /api/export — batch export telemetry data as CSV or XLSX.

Supports multi-channel selection, custom time range, and two output
formats.  Returns a file download response.
"""

from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from . import deps

router = APIRouter()


@router.get("/api/export")
async def api_export(
    channels: str = Query(..., description="Comma-separated channel names, e.g. C-1,D-14"),
    start: float = Query(..., description="Start epoch seconds (inclusive)"),
    end: float = Query(..., description="End epoch seconds (inclusive)"),
    fmt: str = Query("csv", description="Output format: csv or xlsx"),
):
    """Export raw telemetry for one or more channels within a time range.

    Returns a file download with columns: channel, timestamp, raw_value,
    anomaly_score, received_at (ISO 8601).
    """
    c = deps.get()
    ch_list = [ch.strip() for ch in channels.split(",") if ch.strip()]

    # Collect data per channel
    all_rows: list[dict] = []
    for ch in ch_list:
        rows = c.sqlite.query_history(
            channel=ch,
            start_time=start,
            end_time=end,
            limit=100000,
        )
        for r in rows:
            all_rows.append(r)

    # Sort by timestamp
    all_rows.sort(key=lambda r: r["received_at"])

    # Generate filename
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_channels = "_".join(ch_list[:3])
    if len(ch_list) > 3:
        safe_channels += f"_plus{len(ch_list) - 3}"

    if fmt.lower() == "xlsx":
        return _export_xlsx(all_rows, f"telemetry_{safe_channels}_{ts_str}.xlsx")
    else:
        return _export_csv(all_rows, f"telemetry_{safe_channels}_{ts_str}.csv")


def _export_csv(rows: list[dict], filename: str) -> StreamingResponse:
    """Generate a CSV file download response."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["channel", "timestamp", "raw_value", "anomaly_score", "received_at_iso"])

    for r in rows:
        ts = r.get("received_at")
        iso_str = (
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            if ts is not None else ""
        )
        writer.writerow([
            r.get("channel", ""),
            ts if ts is not None else "",
            r.get("raw", ""),
            r.get("score", ""),
            iso_str,
        ])

    content = output.getvalue().encode("utf-8-sig")  # BOM for Excel compatibility
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/csv; charset=utf-8",
        },
    )


def _export_xlsx(rows: list[dict], filename: str) -> StreamingResponse:
    """Generate an XLSX file download response using openpyxl."""
    try:
        from openpyxl import Workbook
    except ImportError:
        # Fallback to CSV if openpyxl is not installed
        return _export_csv(rows, filename.replace(".xlsx", ".csv"))

    wb = Workbook()
    ws = wb.active
    ws.title = "Telemetry"
    ws.append(["channel", "timestamp", "raw_value", "anomaly_score", "received_at_iso"])

    for r in rows:
        ts = r.get("received_at")
        iso_str = (
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            if ts is not None else ""
        )
        ws.append([
            r.get("channel", ""),
            ts if ts is not None else "",
            r.get("raw", ""),
            r.get("score", ""),
            iso_str,
        ])

    # Auto-size columns
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
