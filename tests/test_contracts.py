"""Contract guardrails — 17 invariants from the lessons-learned history.

These tests protect the M1 data path against regressions that were
painful to debug the first time around.  Every invariant cites the
commit where the bug was originally fixed.

Static contracts (source inspection) and dynamic contracts (runtime
verification) live together so a single pytest run covers both.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

# ── paths ──────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
SPACE_DIR = _SRC / "space"
GROUND_DIR = _SRC / "ground"

# ── break the phm.database ↔ phm.services circular import ──────────────
# Importing phm.services.* FIRST fully loads the package, so later
# `from phm.database import RingBuffer` doesn't hit a half-initialised
# module.  (Existing tests under ground/phm/tests/ rely on the same
# trick — see test_phm_layers.py.)
import phm.services  # noqa: F401,E402 — side-effect import


# ════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════

def _lines_of(text: str, needle: str) -> list[int]:
    """Return 0-based line numbers (in ``text``) that contain ``needle``."""
    return [i for i, line in enumerate(text.splitlines()) if needle in line]


def _first_line(text: str, needle: str) -> int:
    """Return 0-based index of first line containing ``needle`` (-1 if absent)."""
    lines = _lines_of(text, needle)
    return lines[0] if lines else -1


def _region(text: str, start_line: int, span: int) -> str:
    """Return a slice of ``span`` lines starting at ``start_line``."""
    return "\n".join(text.splitlines()[start_line : start_line + span])


# ════════════════════════════════════════════════════════════════════════
# C1 — t_acq_start is a top-level JSON field (not buried in metadata)
# ════════════════════════════════════════════════════════════════════════

def test_c1_t_acq_start_is_top_level_field():
    """SpaceServer.enqueue_telemetry pops t_acq_start out of meta and
    writes it as a top-level key on the JSON payload.  Fixed in 7bdf62b.
    """
    src = (SPACE_DIR / "comm.py").read_text(encoding="utf-8")
    assert re.search(r't_acq_start\s*=\s*meta\.pop\(\s*["\']t_acq_start["\']', src), (
        "t_acq_start must be popped out of meta, not left inside metadata"
    )
    assert re.search(r'["\']t_acq_start["\']\s*:\s*t_acq_start', src), (
        "t_acq_start must be written as a top-level key on the payload"
    )


def test_c1_t_acq_start_field_in_ground_packet():
    """GroundClient's TelemetryPacket declares t_acq_start first-class."""
    from comm import TelemetryPacket

    fields = TelemetryPacket.__dataclass_fields__
    assert "t_acq_start" in fields
    assert "metadata" in fields  # still carries metadata dict separately


# ════════════════════════════════════════════════════════════════════════
# C2 — t_acq_start captured immediately after src.read(), before L1/L2
# ════════════════════════════════════════════════════════════════════════

def test_c2_t_acq_captured_before_l1_l2_processing():
    """In space/main.py the call order must be:
       t_acq = time.time() → ipc.read() → L1.check() → L2.detect() → enqueue.

    M1.2 fix: ``t_acq`` must be recorded **before** the IPC request, not after.
    Reason: IPC introduces a network round-trip delay (local ms-level but with
    jitter); if ``t_acq`` is recorded after IPC, the delay pollutes the
    acquisition timestamp and breaks equispaced timestamp reconstruction on
    the ground segment.

    Legacy (pre-M1.2): ``src.read(window)`` → ``t_acq = time.time()``;
    because the local ``read`` is a memory call with negligible latency the
    ordering was less critical.
    """
    src = (SPACE_DIR / "main.py").read_text(encoding="utf-8")
    # Locate the IPC request line (post-M1.2 form)
    ipc_lines = _lines_of(src, "ipc.read(") or _lines_of(src, "ipc_client.read(")
    # Fallback for legacy code: src.read(window)
    if not ipc_lines:
        ipc_lines = _lines_of(src, "src.read(window)")
    t_acq_lines = _lines_of(src, "t_acq = time.time()")
    l1_lines = _lines_of(src, "l1_filter.check(")
    detect_lines = _lines_of(src, "detector.detect(")
    impute_lines = _lines_of(src, "pp._impute(") or _lines_of(src, "pp._impute(raw")

    assert ipc_lines and t_acq_lines, "IPC read or t_acq assignment missing"
    # Use the actual call inside _process_channel (last occurrence)
    ipc_idx = ipc_lines[-1]
    t_acq_idx = t_acq_lines[-1]
    # Key assertion: t_acq is recorded before the IPC request
    assert t_acq_idx < ipc_idx, (
        f"t_acq must come BEFORE ipc.read (line {t_acq_idx+1} vs {ipc_idx+1}) — "
        "otherwise IPC latency leaks into the acquisition timestamp"
    )
    # t_acq must also precede all detection operators
    for name, lines in [("l1_filter.check", l1_lines),
                        ("detector.detect", detect_lines),
                        ("pp._impute", impute_lines)]:
        if lines:
            assert t_acq_idx < lines[-1], (
                f"t_acq must come BEFORE {name} (line {t_acq_idx+1} vs {lines[-1]+1})"
            )


# ════════════════════════════════════════════════════════════════════════
# C3 — L1 SKIP produces zero scores, not None
# ════════════════════════════════════════════════════════════════════════

def test_c3_l1_skip_produces_zeros_not_none():
    """Constant-channel SKIP branch must assign scores = np.zeros(...).
    Returning None caused chart gaps and 'NoneType' crashes.
    """
    src = (SPACE_DIR / "main.py").read_text(encoding="utf-8")
    skip_lines = _lines_of(src, '"skip"')
    assert skip_lines, "'skip' decision branch not found in main.py"
    # Look at the next 12 lines after the first 'skip' mention.
    region = _region(src, skip_lines[0], 12)
    assert "np.zeros" in region, (
        "SKIP branch must assign scores = np.zeros(...) (not None) — checked "
        f"region around line {skip_lines[0]+1}: {region!r}"
    )


# ════════════════════════════════════════════════════════════════════════
# C4 — _ts_quantum == 1.0/sample_rate (full interval, NEVER *2)
# ════════════════════════════════════════════════════════════════════════

def test_c4_ts_quantum_uses_full_sampling_interval():
    """SQLiteStore._ts_quantum must equal 1/sample_rate.  A previous
    version used half the interval + multiplied by 2 in enqueue_predictions,
    which made pred timestamps step at 2× the raw cadence — chart 'holes'.
    Fixed in 7bdf62b.
    """
    src = (_SRC / "ground" / "phm" / "database" / "sqlite_store.py").read_text(encoding="utf-8")
    assert re.search(r"self\._ts_quantum\s*=\s*sample_interval", src), (
        "_ts_quantum must be assigned sample_interval (= 1/sample_rate), not half"
    )


def test_c4_no_doubled_quantum_in_active_code():
    """No active code under ground/phm/ writes _ts_quantum * 2.
    The sqlite_store.py docstring may mention it for historical context.
    """
    offenders = []
    for path in (_SRC / "ground" / "phm").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if "_ts_quantum" in line and "* 2" in line and not stripped.startswith("#"):
                offenders.append(f"{path.relative_to(_SRC)}:{i}: {line.strip()}")
    assert not offenders, (
        "_ts_quantum * 2 found in active code (only allowed in comments): " + "; ".join(offenders)
    )


# ════════════════════════════════════════════════════════════════════════
# C5 — pred timestamps passed explicitly to enqueue_predictions
# ════════════════════════════════════════════════════════════════════════

def test_c5_warning_service_passes_explicit_pred_timestamps():
    """WarningService must pass timestamps=pred_timestamps to
    enqueue_predictions.  Without it, SQLiteStore recomputes via a
    different float path → raw/pred row split.
    """
    src = (_SRC / "ground" / "phm" / "services" / "warning_service.py").read_text(encoding="utf-8")
    assert re.search(r"timestamps\s*=\s*pred_timestamps", src), (
        "warning_service must pass timestamps=pred_timestamps to enqueue_predictions"
    )


# ════════════════════════════════════════════════════════════════════════
# C6 — raw + pred share per-channel table, UPSERT by timestamp PK
# ════════════════════════════════════════════════════════════════════════

def test_c6_telemetry_table_upsert_merges_raw_and_pred():
    """telemetry_<channel> schema must have both raw_value and
    predicted_value columns; both writers use ON CONFLICT(timestamp)
    DO UPDATE so raw and pred UPSERT into the same row.
    """
    src = (_SRC / "ground" / "phm" / "database" / "sqlite_store.py").read_text(encoding="utf-8")
    assert "raw_value" in src and "predicted_value" in src, (
        "telemetry schema must declare both raw_value and predicted_value"
    )
    conflicts = re.findall(r"ON CONFLICT\(timestamp\)\s+DO UPDATE", src)
    assert len(conflicts) >= 2, (
        f"Need ≥2 UPSERT statements (raw writer + pred writer); found {len(conflicts)}"
    )


# ════════════════════════════════════════════════════════════════════════
# C7 — window_size == CONTEXT_LENGTH
# ════════════════════════════════════════════════════════════════════════

def test_c7_window_size_matches_context_length():
    """TSPulse CONTEXT_LENGTH must equal the default window_size in
    DAQ_CONFIG.  Once sensors are parameterised, each must still satisfy
    window == CONTEXT_LENGTH (or a multiple).
    """
    ad_src = (SPACE_DIR / "anomaly_detection.py").read_text(encoding="utf-8")
    m = re.search(r"^CONTEXT_LENGTH\s*=\s*(\d+)", ad_src, re.MULTILINE)
    assert m, "CONTEXT_LENGTH not found in anomaly_detection.py"
    ctx_len = int(m.group(1))

    main_src = (SPACE_DIR / "main.py").read_text(encoding="utf-8")
    m2 = re.search(r'"window_size":\s*(\d+)', main_src)
    if m2:
        win = int(m2.group(1))
        assert win == ctx_len, (
            f"DAQ_CONFIG.window_size ({win}) must equal CONTEXT_LENGTH ({ctx_len})"
        )


# ════════════════════════════════════════════════════════════════════════
# C8 — per-channel table name escaping
# ════════════════════════════════════════════════════════════════════════

def test_c8_table_name_escaping_is_safe():
    """Channel names like 'C-1' must be escaped to 'telemetry_C_1'.
    """
    from phm.database.sqlite_store import SQLiteStore

    # _tel_table is a staticmethod; call without instantiating.
    assert SQLiteStore._tel_table("C-1") == "telemetry_C_1"
    assert SQLiteStore._tel_table("VS-multi_sine") == "telemetry_VS_multi_sine"
    assert SQLiteStore._tel_table("simple") == "telemetry_simple"


# ════════════════════════════════════════════════════════════════════════
# C9 — cross-packet timestamp continuity + _ts_lock serialisation
# ════════════════════════════════════════════════════════════════════════

def test_c9_cross_packet_timestamp_continuity():
    """Two consecutive polls with t_acq_start must produce strictly
    equidistant timestamps (1/sample_rate apart) with no overlap.
    Original zig-zag bug fixed in cf74367 (added _last_ts) + 7bdf62b
    (added t_acq_start anchor).
    """
    import comm as _comm_mod
    from comm import TelemetryPacket
    from phm.database import AlertStore, RingBuffer
    from phm.database.sqlite_store import SQLiteStore
    from phm.services.telemetry_service import TelemetryService
    import time as _time

    sr = 100.0
    n = 512
    t0 = 1_700_000_000.0
    call_state = {"count": 0}

    class _FakeClient:
        """Mimics GroundClient.poll(): returns a list of TelemetryPacket."""
        def __init__(self, *a, **kw):
            pass

        def poll(self, config, timeout=10.0):
            call_state["count"] += 1
            # Second poll's t_acq_start is 5s later (simulating TCP buffer delay).
            t = t0 if call_state["count"] == 1 else t0 + 5.0
            return [TelemetryPacket(
                channel="C-1",
                raw_values=np.arange(n, dtype=np.float32),
                scores=np.zeros(n, dtype=np.float32),
                sample_rate=sr,
                t_acq_start=t,
            )]

    # Patch GroundClient where telemetry_service looks it up.  Note:
    # `from comm import GroundClient` binds the name into the
    # telemetry_service module's namespace, so patching comm.GroundClient
    # alone is NOT enough — we must patch BOTH comm.GroundClient (for any
    # late importers) AND telemetry_service.GroundClient (already bound).
    from phm.services import telemetry_service as _ts_mod
    original_comm = getattr(_comm_mod, "GroundClient", None)
    original_ts = getattr(_ts_mod, "GroundClient", None)
    _comm_mod.GroundClient = _FakeClient
    _ts_mod.GroundClient = _FakeClient
    try:
        ring = RingBuffer()
        alerts = AlertStore()
        sqlite = SQLiteStore(sample_rate=100.0, db_path=":memory:", enabled=True)
        sqlite.start()
        try:
            svc = TelemetryService(ring, alerts, sqlite, space_host="x", space_port=1)
            svc.poll(source_id="test", sample_rate=sr, block_size=n)
            svc.poll(source_id="test", sample_rate=sr, block_size=n)

            # Wait for async batch writer to drain.
            _time.sleep(0.3)

            rows = sqlite._conn.execute(
                'SELECT timestamp FROM "telemetry_C_1" ORDER BY timestamp'
            ).fetchall()
            ts = [r[0] for r in rows]
            assert len(ts) == 2 * n, f"expected {2*n} rows, got {len(ts)}"
            diffs = np.diff(ts)
            # Quantisation introduces tiny float drift (< 1e-6); allow it.
            # The real bug we're guarding against would produce diffs
            # 2× wider (e.g. 0.02 instead of 0.01) or wildly uneven.
            assert np.allclose(diffs, 1.0 / sr, atol=1e-5), (
                f"Timestamps not equidistant at 1/sr. "
                f"diffs range=[{diffs.min()}, {diffs.max()}], "
                f"expected ~{1.0/sr}"
            )
        finally:
            sqlite.close()
    finally:
        if original_comm is not None:
            _comm_mod.GroundClient = original_comm
        if original_ts is not None:
            _ts_mod.GroundClient = original_ts


# ════════════════════════════════════════════════════════════════════════
# C10 — TCP framing: NDJSON + "END\n"
# ════════════════════════════════════════════════════════════════════════

def test_c10_tcp_framing_uses_END_terminator():
    """SpaceServer sends b'END\\n'; GroundClient detects b'\\nEND\\n'.
    """
    space_comm = (SPACE_DIR / "comm.py").read_text(encoding="utf-8")
    ground_comm = (GROUND_DIR / "comm.py").read_text(encoding="utf-8")
    assert 'sendall(b"END\\n")' in space_comm, "SpaceServer must send b'END\\n'"
    assert 'b"\\nEND\\n"' in ground_comm, "GroundClient must detect b'\\nEND\\n'"


# ════════════════════════════════════════════════════════════════════════
# C11 — sample_rate consistency (space DAQ vs ground poller)
# ════════════════════════════════════════════════════════════════════════

def test_c11_sample_rate_consistent_space_and_ground():
    """sample_rate must be consistent across three places (M1.2 update):

    1. ``space/data/space_daq.json`` — DAQ card hardware config (space-side)
    2. ``space/data/signal_sources.json`` — signal-generator default rate
    3. ``ground/django_phm/phm_site/services_bridge.py`` — ground poller literal

    A mismatch silently corrupts the timestamp grid: raw quantised at one
    rate, pred at another, ground reconstructs at a third.

    Pre-M1.2: ``sample_rate`` was hardcoded in ``space/main.py DAQ_CONFIG``.
    Post-M1.2: read from ``space_daq.json``, so the assertion now checks the JSON file.
    """
    import json as _json

    # (1) space_daq.json
    daq_json = _json.loads((SPACE_DIR / "data" / "space_daq.json").read_text(encoding="utf-8"))
    sr_daq = float(daq_json["sample_rate"])

    # (2) signal_sources.json
    sig_json = _json.loads((SPACE_DIR / "data" / "signal_sources.json").read_text(encoding="utf-8"))
    sr_sig = float(sig_json.get("default_sample_rate", sr_daq))

    # (3) services_bridge._poll_one's literal
    bridge = (GROUND_DIR / "django_phm" / "phm_site" / "services_bridge.py").read_text(encoding="utf-8")
    m = re.search(r"telemetry\.poll\(\s*\S+\s*,\s*([\d.]+)\s*,", bridge)
    assert m, (
        "_poll_one's telemetry.poll(...) call not found; expected pattern "
        "`telemetry.poll(src, <rate>, <block>)`"
    )
    sr_ground = float(m.group(1))

    assert sr_daq == sr_sig == sr_ground, (
        f"sample_rate mismatch: space_daq={sr_daq}, signal_sources={sr_sig}, "
        f"ground poller={sr_ground}"
    )


# ════════════════════════════════════════════════════════════════════════
# C12 — AlertPacket carries raw_window + score_window snapshot
# ════════════════════════════════════════════════════════════════════════

def test_c12_alert_packet_has_snapshot_fields():
    from comm import AlertPacket

    fields = AlertPacket.__dataclass_fields__
    assert "raw_window" in fields
    assert "score_window" in fields


def test_c12_space_main_attaches_snapshot_on_alert():
    """space/main.py's alert path must pass raw_window + score_window
    to enqueue_alert.
    """
    src = (SPACE_DIR / "main.py").read_text(encoding="utf-8")
    alert_lines = _lines_of(src, "enqueue_alert")
    assert alert_lines, "enqueue_alert call not found in main.py"
    # Check the multi-line call block (up to 10 lines from first mention).
    region = _region(src, alert_lines[0], 10)
    assert "raw_window" in region and "score_window" in region, (
        "enqueue_alert must pass raw_window= and score_window= snapshots"
    )


# ════════════════════════════════════════════════════════════════════════
# C13 — SpaceServer per-channel buffer + source_id filter
# ════════════════════════════════════════════════════════════════════════

def test_c13_space_server_per_channel_buffer_and_source_filter():
    """SpaceServer must keep per-channel _buffers and a source_id →
    channel map for filtering.  Original single-buffer design drained
    one source's poll at the expense of others.
    """
    src = (SPACE_DIR / "comm.py").read_text(encoding="utf-8")
    assert "_buffers" in src, "SpaceServer must use per-channel _buffers dict"
    assert "register_source" in src, "SpaceServer must expose register_source()"
    assert "_source_map" in src, "SpaceServer must keep _source_map for filtering"


# ════════════════════════════════════════════════════════════════════════
# C14 — TSPulse inference under torch.no_grad()
# ════════════════════════════════════════════════════════════════════════

def test_c14_tspulse_uses_no_grad():
    """Pipeline inference must run inside torch.no_grad().  model.eval()
    alone is NOT enough for thread safety under ThreadPoolExecutor.
    Fixed in 17d3161.
    """
    src = (SPACE_DIR / "anomaly_detection.py").read_text(encoding="utf-8")
    assert "with torch.no_grad()" in src, (
        "AnomalyDetector.detect must wrap pipeline inference in torch.no_grad()"
    )


# ════════════════════════════════════════════════════════════════════════
# C15 — context standardised with same scaler as target
# ════════════════════════════════════════════════════════════════════════

def test_c15_context_uses_same_scaler_as_target():
    """When context is prepended, it must be standardised with the SAME
    scaler fit on the target block (scaler.transform), not a fresh
    fit_transform.  Otherwise short slowly-varying channels score near
    zero (silent missed-detection).
    """
    src = (SPACE_DIR / "anomaly_detection.py").read_text(encoding="utf-8")
    # Context handling must call transform, not fit_transform.
    ctx_lines = [i for i, line in enumerate(src.splitlines()) if "context" in line.lower()]
    assert ctx_lines, "no context handling found in anomaly_detection.py"
    # Check a region around each context mention for transform usage.
    found_transform = False
    for idx in ctx_lines:
        region = _region(src, max(0, idx - 2), 10)
        if "transform" in region and "context" in region:
            found_transform = True
            break
    assert found_transform, (
        "context must be standardised with scaler.transform (the SAME scaler fit "
        "on target block), not a fresh fit_transform"
    )


# ════════════════════════════════════════════════════════════════════════
# C16 — scores np.clip([0,1]); no MinMax normalisation
# ════════════════════════════════════════════════════════════════════════

def test_c16_scores_clipped_no_minmax_normalisation():
    """Pipeline output is StandardScaler'd MSE (no inherent upper bound).
    Must np.clip to [0, 1] (chart Y axis + threshold semantics).

    Per-window MinMax was REMOVED — it forced every window's max to
    1.0, causing false alarms on normal periodic waveforms.
    """
    src = (SPACE_DIR / "anomaly_detection.py").read_text(encoding="utf-8")
    assert "np.clip(scores, 0.0, 1.0)" in src or "np.clip(scores,0.0,1.0)" in src, (
        "scores must be np.clip(scores, 0.0, 1.0)"
    )
    # Active MinMax (not in comments) is forbidden.
    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if "MinMax" in stripped and not stripped.startswith("#"):
            # Allow lines that are clearly comments about WHY not to use it.
            if "removed" in stripped.lower() or "do not" in stripped.lower():
                continue
            pytest.fail(f"Active MinMax normalisation at line {i}: {line.strip()}")


# ════════════════════════════════════════════════════════════════════════
# C17 — ALERT_THRESHOLD space↔ground consistent
# ════════════════════════════════════════════════════════════════════════

def test_c17_alert_threshold_consistent_space_and_ground():
    """The threshold space uses to decide 'score > X → enqueue alert'
    must equal ground's threshold (chart red line + health computation).

    Space side may be ``ALERT_THRESHOLD: float = 0.5`` (with type
    annotation) or ``ALERT_THRESHOLD = 0.5`` — regex accepts both.
    """
    space_main = (SPACE_DIR / "main.py").read_text(encoding="utf-8")
    # Allow optional ": float" type annotation between name and "=".
    m1 = re.search(r"ALERT_THRESHOLD\s*(?::\s*\w+\s*)?=\s*([\d.]+)", space_main)
    assert m1, "ALERT_THRESHOLD not in space/main.py"
    th_space = float(m1.group(1))

    from phm.config import ANOMALY_THRESHOLD

    assert abs(th_space - ANOMALY_THRESHOLD) < 1e-9, (
        f"ALERT_THRESHOLD mismatch: space={th_space}, ground={ANOMALY_THRESHOLD}"
    )


# ════════════════════════════════════════════════════════════════════════
# C18 — IPC server binds 127.0.0.1 only (never exposed externally)
# ════════════════════════════════════════════════════════════════════════

def test_c18_ipc_server_binds_localhost_only():
    """Contract introduced in M1.2: the signal-generator IPC server must bind
    127.0.0.1, never 0.0.0.0.

    IPC is local-only communication between the acquisition card and the signal
    generator. Exposing it would let remote hosts read raw sensor data
    (information leak) or inject forged data (data contamination).
    """
    ipc_src = (SPACE_DIR / "signal_ipc.py").read_text(encoding="utf-8")
    # IPC_HOST constant must be 127.0.0.1
    assert re.search(r'^IPC_HOST\s*=\s*["\']127\.0\.0\.1["\']', ipc_src, re.MULTILINE), (
        "IPC_HOST must be '127.0.0.1' (never '0.0.0.0' or other)"
    )
    # No 0.0.0.0 may appear as an actual bind/host parameter.
    # Only inspect lines that look like code: indented, not inside a quoted string literal.
    # 0.0.0.0 inside docstrings, comments, or prose is allowed.
    bad = []
    for i, line in enumerate(ipc_src.splitlines(), 1):
        stripped = line.strip()
        if "0.0.0.0" not in stripped:
            continue
        if stripped.startswith("#"):
            continue  # comment
        if '"""' in line or "'''" in line:
            continue  # docstring delimiter line
        # Skip lines that are obviously inside a docstring (heuristic:
        # line has CJK chars or starts with prose — those are docstring body)
        if any('\u4e00' <= ch <= '\u9fff' for ch in stripped):
            continue  # Chinese line, likely a docstring
        # Active code line containing 0.0.0.0 — flag it.
        bad.append(f"line {i}: {stripped}")
    assert not bad, (
        "0.0.0.0 found in signal_ipc.py active code (only allowed in comments/"
        "docstrings): " + "; ".join(bad)
    )


def test_c18_ipc_default_port_documented():
    """IPC default port 9878 should be declared as a constant in signal_ipc.py for centralized management."""
    ipc_src = (SPACE_DIR / "signal_ipc.py").read_text(encoding="utf-8")
    assert re.search(r"^IPC_DEFAULT_PORT\s*=\s*\d+", ipc_src, re.MULTILINE), (
        "IPC_DEFAULT_PORT constant not defined in signal_ipc.py"
    )


# ════════════════════════════════════════════════════════════════════════
# C19 — AlertPacket carries acq_ts (true anomaly sampling timestamp)
# ════════════════════════════════════════════════════════════════════════
# Day22 issue 3.3b: the front-end red dot was offset from the real anomaly
# position because AlertPacket only carried wall-clock time.time() (ground
# receipt time), which is a different clock from the telemetry sampling grid
# (anchored by `t_acq_start`). The fix adds
# `acq_ts = t_acq + argmax(scores)/sr` propagated through the whole chain
# (space → ground → sqlite.created_at → alert_points API → front-end tsToX).

def test_c19_alert_packet_has_acq_ts_field():
    """AlertPacket must contain the acq_ts field (true anomaly sampling timestamp)."""
    from comm import AlertPacket
    fields = AlertPacket.__dataclass_fields__
    assert "acq_ts" in fields, "AlertPacket must have acq_ts field"


def test_c19_space_main_computes_acq_ts_from_argmax():
    """The alert path in space/main.py must use nanargmax to compute the peak position and pass acq_ts."""
    src = (SPACE_DIR / "main.py").read_text(encoding="utf-8")
    alert_lines = _lines_of(src, "enqueue_alert")
    assert alert_lines, "enqueue_alert call not found in main.py"
    region = _region(src, alert_lines[0] - 15, 20)  # look backward for the acq_ts computation logic
    assert "acq_ts" in region, "enqueue_alert must pass acq_ts="
    assert "nanargmax" in region or "argmax" in region, (
        "acq_ts must be computed from argmax/nanargmax of scores"
    )


def test_c19_ground_client_poll_reads_acq_ts():
    """GroundClient.poll must read the acq_ts field when parsing an alert packet."""
    src = (GROUND_DIR / "comm.py").read_text(encoding="utf-8")
    # Locate the code block where AlertPacket is constructed
    idx = _first_line(src, 'obj.get("type") == "alert"')
    assert idx >= 0, "alert parsing branch not found in ground/comm.py"
    region = _region(src, idx, 12)
    assert "acq_ts" in region, (
        "GroundClient.poll must read acq_ts when constructing AlertPacket"
    )


def test_c19_telemetry_service_prefers_acq_ts():
    """TelemetryService must prefer `AlertPacket.acq_ts` as the alert timestamp,
    rather than wall-clock time.time()."""
    src = (GROUND_DIR / "phm" / "services" / "telemetry_service.py").read_text(encoding="utf-8")
    idx = _first_line(src, "isinstance(p, AlertPacket)")
    assert idx >= 0, "AlertPacket handling not found in telemetry_service.py"
    region = _region(src, idx, 10)
    assert "acq_ts" in region, (
        "telemetry_service must use p.acq_ts as alert time (fallback time.time())"
    )
