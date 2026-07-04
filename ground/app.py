"""Ground-segment Streamlit application.

Run independently from the space segment::

    streamlit run src/ground/app.py

Connects to the space TCP server (localhost:9876 by default), polls for
telemetry/alert data, and renders real-time charts.  Settings are persisted
in ``src/ground/settings.json``.

Requires the space segment to be running first::

    python src/space/main.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Path & environment
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_HERE.parent / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from i18n import t, LANGS
from comm import GroundClient, TelemetryPacket, AlertPacket

SETTINGS_PATH = _HERE / "settings.json"

# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------
DEFAULTS = {
    "lang": "zh",
    "source_id": "file:NASA-MSL/C-1",
    "use_detection": True,
    "use_forecast": True,
    "window_size": 512,
    "sample_rate": 50.0,
    "space_host": "127.0.0.1",
    "space_port": 9876,
}


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return {**DEFAULTS, **json.loads(SETTINGS_PATH.read_text("utf-8"))}
        except Exception:
            return dict(DEFAULTS)
    return dict(DEFAULTS)


def save_settings(d: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Space Payload Health Monitor", page_icon="🛰️", layout="wide")

cfg = load_settings()
if "lang" not in st.session_state:
    st.session_state["lang"] = cfg["lang"]
lang = st.session_state["lang"]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header(t("config", lang))

    lc = st.selectbox(t("lang_label", lang), list(LANGS.keys()),
                      format_func=lambda k: LANGS[k],
                      index=list(LANGS.keys()).index(lang),
                      key="lang_choice")
    if lc != st.session_state["lang"]:
        st.session_state["lang"] = lc
        cfg["lang"] = lc
        save_settings(cfg)
        st.rerun()
    lang = st.session_state["lang"]

    # ---- data source (unified dropdown) ----
    st.subheader("📡 " + ("数据源" if lang == "zh" else "Data Source"))

    # Scan all available sources
    try:
        import sys as _sys
        _space_dir = str(_HERE.parent / "space")
        if _space_dir not in _sys.path:
            _sys.path.insert(0, _space_dir)
        from sensor_source import list_all_sources
        all_sources = list_all_sources()
    except Exception:
        all_sources = []

    source_options = {s["id"]: s["label"] for s in all_sources}
    default_source = cfg.get("source_id", "file:NASA-MSL/C-1")
    if default_source not in source_options and all_sources:
        default_source = all_sources[0]["id"]

    source_id = st.selectbox(
        "来源" if lang == "zh" else "Source",
        list(source_options.keys()),
        format_func=lambda sid: source_options.get(sid, sid),
        index=list(source_options.keys()).index(default_source)
        if default_source in source_options else 0,
        key="source_id",
        disabled=st.session_state.get("playing", False),  # lock during playback
    )

    # When source changes while paused, clear buffer and state.
    # The next poll cycle (0.5s) sends the new source_id to space.
    if "active_source_id" not in st.session_state:
        st.session_state["active_source_id"] = source_id
    if st.session_state["active_source_id"] != source_id:
        st.session_state["active_source_id"] = source_id
        st.session_state["telemetry_ring"] = []
        st.session_state["alert_list"] = []
        st.session_state["display_pos"] = 0
        st.session_state["source_exhausted"] = False
        st.session_state["fc_result"] = None
        st.session_state["warn_events"] = []
        st.session_state["playing"] = False

    # Show lock hint when playing
    if st.session_state.get("playing", False):
        st.caption("⏸️ " + ("暂停后可切换来源" if lang == "zh" else "Pause to switch source"))

    # ---- models ----
    st.subheader("🧠 " + ("模型" if lang == "zh" else "Models"))
    use_detection = st.checkbox(
        "TSPulse " + ("检测" if lang == "zh" else "Detection"),
        value=cfg.get("use_detection", True), key="use_detection",
    )
    use_forecast = st.checkbox(
        "TTM-R3 " + ("预测" if lang == "zh" else "Forecast"),
        value=cfg.get("use_forecast", True), key="use_forecast",
    )

    # ---- connection ----
    with st.expander(
        "🌐 " + ("连接设置" if lang == "zh" else "Connection"), expanded=False,
    ):
        space_host = st.text_input(
            "主机" if lang == "zh" else "Host",
            cfg.get("space_host", "127.0.0.1"), key="space_host",
        )
        space_port = st.number_input(
            "端口" if lang == "zh" else "Port",
            value=cfg.get("space_port", 9876), min_value=1, max_value=65535,
            key="space_port",
        )

    # ---- window + rate (must be before Save button — it references them) ----
    window_size = st.select_slider(
        "窗口" if lang == "zh" else "Window",
        options=[256, 512, 1024],
        value=cfg.get("window_size", 512), key="window_sel",
    )
    st.session_state["display_window"] = window_size
    sample_rate = st.number_input(
        "采样率 Hz" if lang == "zh" else "Sample rate Hz",
        value=cfg.get("sample_rate", 50.0), min_value=-1.0, step=1.0, key="sample_rate",
    )

    # ---- save ----
    if st.button("💾 " + ("保存设置" if lang == "zh" else "Save Settings")):
        save_settings({
            "lang": st.session_state.get("lang", "zh"),
            "source_id": source_id,
            "use_detection": use_detection,
            "use_forecast": use_forecast,
            "space_host": space_host,
            "space_port": space_port,
            "window_size": window_size,
            "sample_rate": sample_rate,
        })
        st.success("✅ " + ("已保存" if lang == "zh" else "Saved"))


# ---------------------------------------------------------------------------
# Session state init — ring buffer & playback
# ---------------------------------------------------------------------------
# telemetry_ring: deque of {"raw": float, "score": float|None, "step": int}
if "telemetry_ring" not in st.session_state:
    st.session_state["telemetry_ring"] = []  # list used as ring buffer
if "alert_list" not in st.session_state:
    st.session_state["alert_list"] = []  # accumulating alerts
if "space_connected" not in st.session_state:
    st.session_state["space_connected"] = False
if "playing" not in st.session_state:
    st.session_state["playing"] = False
if "source_exhausted" not in st.session_state:
    st.session_state["source_exhausted"] = False
if "display_pos" not in st.session_state:
    st.session_state["display_pos"] = 0  # right edge of display window
if "fc_result" not in st.session_state:
    st.session_state["fc_result"] = None  # {"ctx": [...], "pred": [...], "anchor_pos": int}
if "warn_events" not in st.session_state:
    st.session_state["warn_events"] = []  # [{anchor, pred_idx, score, time, false_alarm}]

PLAYBACK_BUFFER_SIZE = 20000  # max samples in ring buffer
DEFAULT_DISPLAY_WINDOW = 512   # visible samples on oscilloscope


# ---------------------------------------------------------------------------
# Cached models — loaded lazily only when needed
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_forecaster():
    from forecasting import TrendForecaster
    return TrendForecaster(device="cpu")


@st.cache_resource
def _load_detector():
    import sys, os
    _space_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "space")
    if _space_dir not in sys.path:
        sys.path.insert(0, _space_dir)
    from anomaly_detection import AnomalyDetector
    return AnomalyDetector(device="cpu")


# ---------------------------------------------------------------------------
# Helper: convert ring-buffer index to elapsed time string (mm:ss.t)
# ---------------------------------------------------------------------------
def _idx_to_time_str(idx: int, sr: float, t0: float | None = None) -> str:
    """Format ring-buffer index as wall-clock time.

    Sub-second precision adapts to sample rate so individual points are
    distinguishable: whole seconds for <=1 Hz, centiseconds for <=100 Hz,
    milliseconds for >100 Hz.
    """
    if t0 is not None and sr > 0:
        ts = t0 + idx / sr
        if sr <= 1:
            return time.strftime("%H:%M:%S", time.localtime(ts))
        elif sr <= 100:
            return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int(ts * 100) % 100:02d}"
        else:
            return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int(ts * 1000) % 1000:03d}"
    if sr <= 0:
        return str(idx)
    elapsed = idx / sr
    minutes = int(elapsed) // 60
    seconds = elapsed % 60
    return f"{minutes:02d}:{seconds:06.3f}"


# ---------------------------------------------------------------------------


@st.fragment(run_every=0.5)
def _dynamic_content():
    # Poll space segment — every cycle (keeps connection alive, handles
    # source-switch reconfig immediately).  When paused, data is DISCARDED
    # (freeze accumulation); when playing, data is stored.
    # ---------------------------------------------------------------------------
    send_cfg = {
        "source_id": source_id,
        "sample_rate": sample_rate,
        "use_detection": use_detection,
    }

    packets: list = []
    try:
        client = GroundClient(host=space_host, port=space_port, timeout=1.5)
        _all = client.poll(send_cfg)
        if _all:
            st.session_state["space_connected"] = True
        if st.session_state.get("playing", False):
            packets = _all
    except Exception:
        pass

    # Merge new packets into ring buffer.
    # Each sample gets its own wall-clock time, distributed linearly across
    # the batch so the X-axis aligns with real time (same clock as alerts).
    _pkt_time = time.time()
    _pkt_sr = sample_rate if sample_rate > 0 else 1.0
    new_raw_count = 0
    for p in packets:
        if isinstance(p, TelemetryPacket):
            raw = p.raw_values
            scores = p.scores
            step_base = p.metadata.get("step", 0) - len(raw)
            n = len(raw)
            for i in range(n):
                # Last sample (i=n-1) ≈ now; first sample ≈ n/_pkt_sr seconds ago
                sample_time = _pkt_time - (n - 1 - i) / _pkt_sr
                st.session_state["telemetry_ring"].append({
                    "raw": float(raw[i]),
                    "score": float(scores[i]) if scores is not None and i < len(scores) else None,
                    "step": step_base + i + 1,
                    "received_at": sample_time,
                })
                new_raw_count += 1
            if p.metadata.get("exhausted", False):
                st.session_state["source_exhausted"] = True
        elif isinstance(p, AlertPacket):
            st.session_state["alert_list"].append({
                "channel": p.channel,
                "score": p.score,
                "step": p.step,
                "message": p.message,
                "time": time.time(),
            })

    # Trim ring buffer to max size
    ring = st.session_state["telemetry_ring"]
    if len(ring) > PLAYBACK_BUFFER_SIZE:
        overflow = len(ring) - PLAYBACK_BUFFER_SIZE
        st.session_state["telemetry_ring"] = ring[overflow:]
        # Adjust display_pos to stay within bounds
        if st.session_state["display_pos"] < overflow:
            st.session_state["display_pos"] = 0
        else:
            st.session_state["display_pos"] -= overflow

    # Advance display position when playing — BATCH-BASED, not smooth scroll.
    # Real telemetry arrives in discrete batches; the display jumps to the
    # latest data on each batch arrival, matching real ground-station behaviour.
    if st.session_state["playing"] and new_raw_count > 0:
        st.session_state["display_pos"] = len(ring)

    # Auto-pause when exhausted and reached the end
    if st.session_state["source_exhausted"] and st.session_state["playing"]:
        if st.session_state["display_pos"] >= len(st.session_state["telemetry_ring"]):
            st.session_state["playing"] = False

    # ---- auto-forecast: on each new batch, TTM-R3 predicts 96 steps,
    #      then TSPulse detects anomalies on the prediction = 预警.
    #      This end-to-end foundation-model cascade is the core innovation.
    #      (3σ baseline is kept as a comparison method for the paper.)
    if st.session_state["playing"] and new_raw_count > 0 and len(ring) >= 512:
        _prev_fc = st.session_state.get("fc_result")
        _need_forecast = (_prev_fc is None or
                          st.session_state["display_pos"] >= _prev_fc.get("anchor_pos", 0))
        if _need_forecast:
            _ctx_end = min(len(ring), st.session_state["display_pos"])
            _ctx_start = max(0, _ctx_end - 512)
            if _ctx_end - _ctx_start >= 512:
                try:
                    _fc = _load_forecaster()
                    _ctx_raw = np.array(
                        [ring[i]["raw"] for i in range(_ctx_start, _ctx_end)],
                        dtype=np.float32)
                    _ctx, _pred, _scl = _fc.forecast(_ctx_raw)
                    # ---- TSPulse cascade: detect anomalies in prediction ----
                    _combined = np.concatenate([
                        _ctx[-416:].astype(np.float32),
                        _pred.astype(np.float32)])
                    try:
                        _det = _load_detector()
                        _warn = _det.detect(_combined)
                        _warn = _warn[-96:]
                    except Exception:
                        _warn = np.zeros(96, dtype=np.float32)
                    st.session_state["fc_result"] = {
                        "ctx": _ctx.tolist() if hasattr(_ctx, "tolist") else list(_ctx),
                        "pred": _pred.tolist() if hasattr(_pred, "tolist") else list(_pred),
                        "anchor_pos": _ctx_end,
                        "warn_scores": _warn.tolist() if hasattr(_warn, "tolist") else list(_warn),
                    }
                    # Store warning events
                    _warn_events = st.session_state.get("warn_events", [])
                    for _i, _s in enumerate(_warn):
                        if float(_s) > 0.5:
                            _warn_events.append({
                                "anchor": _ctx_end,
                                "pred_idx": _i,
                                "score": float(_s),
                                "time": time.time(),
                                "false_alarm": False,
                            })
                    st.session_state["warn_events"] = _warn_events[-200:]
                except Exception:
                    pass

    # ---- false-alarm check: when display advances past a prediction anchor,
    #      compare predicted warnings with real alerts ----
    _display_pos = st.session_state["display_pos"]
    for _we in st.session_state.get("warn_events", []):
        if not _we["false_alarm"] and _display_pos >= _we["anchor"] + _we["pred_idx"]:
            # Real data is now available — check if it triggered a real alert
            _real_idx = _we["anchor"] + _we["pred_idx"]
            if _real_idx < len(ring):
                _real_score = ring[_real_idx].get("score")
                if _real_score is None or _real_score < 0.5:
                    _we["false_alarm"] = True


    # ---------------------------------------------------------------------------
    # Title bar + control bar
    # ---------------------------------------------------------------------------
    st.title(t("title", lang))
    st.caption(t("subtitle", lang))

    ring = st.session_state["telemetry_ring"]
    total_samples = len(ring)
    conn_ok = st.session_state.get("space_connected", False) or total_samples > 0
    st.markdown(
        f"<strong>{'🟢' if conn_ok else '🔴'} {'天基已连接' if conn_ok else '等待天基…'}</strong> | "
        f"tcp://{space_host}:{space_port} | "
        f"{'总样本' if lang == 'zh' else 'Samples'}: {total_samples}"
        + (f" | {'⚠️ 已停止' if lang == 'zh' else '⚠️ Stopped'}"
           if st.session_state["source_exhausted"] else ""),
        unsafe_allow_html=True,
    )

    # ---- control bar (play / pause / reset) ----
    ctrl_cols = st.columns([1, 1, 1, 4])
    with ctrl_cols[0]:
        if st.session_state["playing"]:
            if st.button("⏸️ " + ("暂停" if lang == "zh" else "Pause"), width='stretch'):
                st.session_state["playing"] = False
                st.rerun()
        else:
            disabled = st.session_state["source_exhausted"] and \
                st.session_state["display_pos"] >= total_samples
            if st.button("▶️ " + ("播放" if lang == "zh" else "Play"),
                         width='stretch', type="primary",
                         disabled=disabled):
                st.session_state["playing"] = True
                # If display_pos is at 0 but we have buffered data, show latest
                if st.session_state["display_pos"] == 0 and total_samples > 0:
                    _dw = st.session_state.get("display_window", DEFAULT_DISPLAY_WINDOW)
                    st.session_state["display_pos"] = min(_dw, total_samples)
                # If at the end, reset to latest
                if st.session_state["display_pos"] >= total_samples:
                    st.session_state["display_pos"] = total_samples
                st.rerun()
    with ctrl_cols[1]:
        if st.button("🔄 " + ("重置" if lang == "zh" else "Reset"), width='stretch'):
            st.session_state["playing"] = False
            st.session_state["display_pos"] = 0
            st.session_state["telemetry_ring"] = []
            st.session_state["alert_list"] = []
            # Keep space_connected = True — the space segment IS still online.
            # Only clear data-related state.
            st.session_state["fc_result"] = None
            st.session_state["warn_events"] = []
            st.rerun()
    with ctrl_cols[2]:
        dw = st.session_state.get("display_window", DEFAULT_DISPLAY_WINDOW)
        st.caption(f"📏 {dw} " + ("点" if lang == "zh" else "pts"))
    with ctrl_cols[3]:
        display_pos = st.session_state["display_pos"]
        if total_samples > 0:
            progress = min(display_pos / total_samples, 1.0)
            st.progress(progress, text=f"⏱️ {display_pos}/{total_samples}"
                        if lang == "zh" else f"⏱️ Step {display_pos}/{total_samples}")


    # ---------------------------------------------------------------------------
    # Section 1 — Oscilloscope telemetry display
    # ---------------------------------------------------------------------------
    st.header("📡 " + ("遥测信号" if lang == "zh" else "Telemetry Signal"))

    ring = st.session_state["telemetry_ring"]
    display_pos = st.session_state["display_pos"]
    display_window = st.session_state.get("display_window", DEFAULT_DISPLAY_WINDOW)

    if not ring:
        if st.session_state.get("playing", False):
            st.info("⏳ " + ("加载中…" if lang == "zh" else "Loading…"))
        elif conn_ok:
            st.info("已连接，点击 ▶️ 播放开始接收数据" if lang == "zh"
                    else "Connected — click ▶️ Play to start receiving")
        else:
            st.info("⏳ " + ("等待天基数据 — 请先启动 python src/space/main.py" if lang == "zh"
                             else "Waiting — start space segment first"))
    else:
        # ---- paused: position slider to scroll through history ----
        # on_change callback gives immediate refresh while dragging (no need to
        # release the mouse first).
        def _on_pos_change():
            st.session_state["display_pos"] = st.session_state["pos_slider"]

        if not st.session_state["playing"] and len(ring) > display_window:
            st.slider(
                "位置" if lang == "zh" else "Position",
                0, len(ring),
                value=min(display_pos, len(ring)),
                step=1,
                key="pos_slider",
                on_change=_on_pos_change,
            )
            display_pos = st.session_state["display_pos"]

        # Extract visible window
        left = max(0, display_pos - display_window)
        right = display_pos
        visible = ring[left:right]

        if len(visible) == 0:
            st.info("拖动位置滑块查看数据" if lang == "zh" else "Drag the position slider to view data")
        else:
            raw_arr = np.array([v["raw"] for v in visible], dtype=np.float32)
            x_num = np.arange(left, right, dtype=np.int32)
            _sr_chart = sample_rate if sample_rate > 0 else 1.0
            # Wall-clock t0: back-compute from first visible sample's receive time
            _t0 = visible[0].get("received_at", None)
            if _t0 is not None:
                _t0 = _t0 - left / _sr_chart  # time of sample index 0
            # Time string for EVERY visible point (used in hover tooltip)
            hover_time = np.array([_idx_to_time_str(i, _sr_chart, _t0) for i in range(left, right)])
            # ~6 tick marks evenly spaced
            n_ticks = min(6, right - left)
            if n_ticks > 1:
                tick_step = max(1, (right - left) // (n_ticks - 1))
                tick_positions = list(range(left, right, tick_step))
            else:
                tick_positions = [left]
            tick_labels = [_idx_to_time_str(t, _sr_chart, _t0) for t in tick_positions]

            # Always build dual-panel chart (scores = 0 where unavailable)
            scores_arr = np.array(
                [v["score"] if v["score"] is not None else 0.0 for v in visible],
                dtype=np.float32,
            )
            if all(s == 0.0 for s in scores_arr):
                st.caption("⚠️ " + ("当前窗口无检测分数" if lang == "zh" else "No detection scores in view"))

            # Static legend (outside chart to avoid flicker on redraw)
            st.caption("🔵 " + ("遥测值" if lang == "zh" else "Value") +
                       "  |  🟠 " + ("异常分数" if lang == "zh" else "Score") +
                       "  |  — — " + ("阈值 0.5" if lang == "zh" else "Threshold 0.5"))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=(
                                    "遥测信号" if lang == "zh" else "Telemetry",
                                    "异常分数 (TSPulse)" if lang == "zh" else "Anomaly Score",
                                ),
                                row_heights=[0.55, 0.45])
            fig.add_trace(go.Scatter(x=x_num, y=raw_arr, mode="lines",
                                     line=dict(color="#1f77b4", width=1),
                                     showlegend=False,
                                     customdata=hover_time,
                                     hovertemplate="时间: %{customdata}<br>值: %{y:.4f}<extra></extra>"),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=x_num, y=scores_arr, mode="lines",
                                     line=dict(color="#ff7f0e", width=1.5),
                                     fill="tozeroy", fillcolor="rgba(255,127,14,0.12)",
                                     showlegend=False,
                                     customdata=hover_time,
                                     hovertemplate="时间: %{customdata}<br>分数: %{y:.4f}<extra></extra>"),
                          row=2, col=1)

            # Threshold line at 0.5
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.4, row=2, col=1)

            # ---- prediction overlay (auto-forecast while playing) ----
            fc = st.session_state.get("fc_result")
            if fc is not None:
                ctx_arr = np.array(fc["ctx"], dtype=np.float32)
                pred_arr = np.array(fc["pred"], dtype=np.float32)
                anchor = fc["anchor_pos"]
                ctx_x = np.arange(anchor - len(ctx_arr), anchor, dtype=np.int32)
                pred_x = np.arange(anchor, anchor + len(pred_arr), dtype=np.int32)
                fig.add_trace(go.Scatter(x=ctx_x, y=ctx_arr, mode="lines",
                                         line=dict(color="#2ca02c", width=1, dash="dot"),
                                         opacity=0.5, showlegend=False),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=pred_x, y=pred_arr, mode="lines",
                                         line=dict(color="#2ca02c", width=2, dash="dash"),
                                         showlegend=False),
                              row=1, col=1)
                # Warning markers on prediction
                warn_s = np.array(fc.get("warn_scores", []), dtype=np.float32)
                if len(warn_s) == len(pred_x):
                    wm = warn_s > 0.5
                    if wm.any():
                        fig.add_trace(go.Scatter(
                            x=pred_x[wm], y=pred_arr[wm],
                            mode="markers",
                            marker=dict(color="red", size=4, symbol="x"),
                            showlegend=False,
                            hovertemplate="⚠️ 预警<extra></extra>",
                        ), row=1, col=1)
                fig.add_vrect(x0=anchor, x1=anchor + len(pred_arr),
                              fillcolor="rgba(44,160,44,0.06)", row=1, col=1)

            fig.update_layout(height=420, hovermode="x unified", font=dict(size=12),
                              margin=dict(l=20, r=20, t=30, b=20),
                              uirevision="true")
            # Time-formatted tick labels on bottom subplot
            fig.update_xaxes(title_text="时间" if lang == "zh" else "Time",
                             tickvals=tick_positions, ticktext=tick_labels, row=2, col=1)
            # Lock x-axis range to maintain stable viewport (oscilloscope effect)
            if st.session_state["playing"]:
                fig.update_xaxes(range=[left, right], row=1, col=1)
                fig.update_xaxes(range=[left, right], row=2, col=1)
                if st.session_state["playing"]:
                    fig.update_xaxes(range=[left, right])

            st.plotly_chart(fig, width='stretch', key="oscilloscope")


    # ---------------------------------------------------------------------------
    # Section 2 — Alerts (accumulating)
    # ---------------------------------------------------------------------------
    st.header("🚨 " + ("告警" if lang == "zh" else "Alerts"))
    alert_list = st.session_state.get("alert_list", [])
    if alert_list:
        with st.expander(
            f"📋 {len(alert_list)} " + ("条告警" if lang == "zh" else "alerts"),
            expanded=len(alert_list) <= 5,
        ):
            for a in reversed(alert_list[-50:]):
                ts = time.strftime("%H:%M:%S", time.localtime(a["time"]))
                st.error(
                    f"🕐 `{ts}` | ⚠️ `{a['channel']}` | "
                    f"step={a['step']} | score={a['score']:.4f}"
                )
    else:
        st.success("✅ " + ("当前无告警" if lang == "zh" else "No alerts"))


    # ---------------------------------------------------------------------------
    # Section 3 — 预警 (forecast-based early warnings)
    # ---------------------------------------------------------------------------
    st.header("⚠️ " + ("预警（TTM-R3 预测 + TSPulse 检测）" if lang == "zh"
                       else "Early Warning (TTM-R3 + TSPulse)"))
    warn_events = st.session_state.get("warn_events", [])
    if not use_forecast:
        st.caption("预测未启用" if lang == "zh" else "Forecasting disabled")
    elif not warn_events:
        st.success("✅ " + ("当前无预警" if lang == "zh" else "No early warnings"))
    else:
        # Show latest 20 warnings
        n_false = sum(1 for w in warn_events if w["false_alarm"])
        with st.expander(
            f"📋 {len(warn_events)} " + ("条预警" if lang == "zh" else "warnings") +
            (f"（{n_false} " + ("条误报" if lang == "zh" else "false alarms") + "）"
             if n_false > 0 else ""),
            expanded=len(warn_events) <= 5,
        ):
            for w in reversed(warn_events[-20:]):
                _ts = time.strftime("%H:%M:%S", time.localtime(w["time"]))
                _tag = " 🟡 [误报]" if w["false_alarm"] else ""
                st.warning(
                    f"🕐 `{_ts}` | anchor={w['anchor']} | "
                    f"pred_idx={w['pred_idx']} | score={w['score']:.4f}{_tag}"
                )


    # ---------------------------------------------------------------------------
    # Footer & auto-refresh
    # ---------------------------------------------------------------------------
    st.divider()
    st.caption(t("footer", lang))

    c1, c2 = st.columns(2)
    c1.caption(f"Ring buffer: {len(ring)} samples | Alerts: {len(alert_list)}")
    status = "▶️ Playing" if st.session_state.get("playing") else "⏸️ Paused"
    c2.caption(f"{status} | Polling every 0.5s")





_dynamic_content()
