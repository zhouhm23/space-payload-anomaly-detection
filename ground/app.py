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
    "source_type": "dataset",
    "dataset_name": "NASA-MSL",
    "channel": "C-1",
    "signal_type": "multi_sine",
    "freq": 0.02,
    "noise_enabled": False,
    "missing_rate": 0.05,
    "noise_std": 0.08,
    "jitter_std": 0.5,
    "use_detection": True,
    "use_forecast": True,
    "window_size": 512,
    "sample_rate": 1.0,
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

    # ---- data source ----
    st.subheader("📡 " + ("数据源" if lang == "zh" else "Data Source"))
    source_type = st.radio(
        "来源" if lang == "zh" else "Source",
        ["dataset", "synthetic"],
        format_func=lambda x: "📂 NASA 数据集" if x == "dataset" else "🎛️ 合成信号",
        index=0 if cfg.get("source_type") == "dataset" else 1,
        key="source_type",
    )

    # source-specific controls
    dataset_name = None
    channel = None
    signal_type = None
    freq = None
    noise_enabled = False
    missing_rate = 0.0
    noise_std = 0.0
    jitter_std = 0.0

    if source_type == "dataset":
        dataset_name = st.selectbox(
            "Dataset", ["NASA-MSL", "NASA-SMAP"],
            index=0 if cfg.get("dataset_name", "NASA-MSL") == "NASA-MSL" else 1,
            key="dataset_name",
        )
        from data_loader import list_channels
        chs = list_channels(dataset_name)
        ch_names = [c[0] for c in chs]
        default_ch = cfg.get("channel", "C-1")
        ch_idx = ch_names.index(default_ch) if default_ch in ch_names else 0
        channel = st.selectbox("Channel", ch_names, index=ch_idx, key="channel_sel")
    else:
        signal_type = st.selectbox(
            "信号类型" if lang == "zh" else "Signal Type",
            ["multi_sine", "sine", "square", "chirp"],
            index=["multi_sine", "sine", "square", "chirp"].index(
                cfg.get("signal_type", "multi_sine")),
            key="signal_type",
        )
        freq = st.slider(
            "频率" if lang == "zh" else "Frequency",
            0.001, 0.1, cfg.get("freq", 0.02), 0.001, key="freq",
        )
        # noise settings only for synthetic signals
        with st.expander(
            "🔧 " + ("噪声设置" if lang == "zh" else "Noise"), expanded=False,
        ):
            noise_enabled = st.checkbox(
                "启用" if lang == "zh" else "Enable",
                value=cfg.get("noise_enabled", False), key="noise_enabled",
            )
            if noise_enabled:
                missing_rate = st.slider(
                    "缺失率" if lang == "zh" else "Missing rate",
                    0.0, 0.2, cfg.get("missing_rate", 0.05), 0.01, key="missing_rate",
                )
                noise_std = st.slider(
                    "噪声 σ" if lang == "zh" else "Noise σ",
                    0.0, 0.3, cfg.get("noise_std", 0.08), 0.01, key="noise_std",
                )
                jitter_std = st.slider(
                    "抖动" if lang == "zh" else "Jitter",
                    0.0, 2.0, cfg.get("jitter_std", 0.5), 0.1, key="jitter_std",
                )

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
    sample_rate = st.number_input(
        "采样率 Hz" if lang == "zh" else "Sample rate Hz",
        value=cfg.get("sample_rate", 1.0), min_value=-1.0, step=1.0, key="sample_rate",
    )

    # ---- save ----
    if st.button("💾 " + ("保存设置" if lang == "zh" else "Save Settings")):
        save_settings({
            "lang": st.session_state.get("lang", "zh"),
            "source_type": source_type,
            "dataset_name": dataset_name,
            "channel": channel,
            "signal_type": signal_type,
            "freq": freq,
            "noise_enabled": noise_enabled,
            "missing_rate": missing_rate,
            "noise_std": noise_std,
            "jitter_std": jitter_std,
            "use_detection": use_detection,
            "use_forecast": use_forecast,
            "space_host": space_host,
            "space_port": space_port,
            "window_size": window_size,
            "sample_rate": sample_rate,
        })
        st.success("✅ " + ("已保存" if lang == "zh" else "Saved"))


# ---------------------------------------------------------------------------
# Cached models — loaded lazily only when needed (not at page render)
# ---------------------------------------------------------------------------
# NOTE: Anomaly detection runs on the SPACE segment; the ground segment only
# does forecasting (triggered by the user clicking "Run Forecast").  We do
# NOT preload any model here — that caused a 30s blank page on startup.

@st.cache_resource
def _load_forecaster():
    from forecasting import TrendForecaster
    return TrendForecaster(device="cpu")


# ---------------------------------------------------------------------------
# Poll space segment — send current config, receive data
# ---------------------------------------------------------------------------
send_cfg = {
    "source_type": source_type,
    "dataset_name": dataset_name,
    "channel": channel,
    "signal_type": signal_type,
    "freq": freq,
    "noise_enabled": noise_enabled,
    "missing_rate": missing_rate,
    "noise_std": noise_std,
    "jitter_std": jitter_std,
    "sample_rate": sample_rate,
    "use_detection": use_detection,
}
# Strip None values
send_cfg = {k: v for k, v in send_cfg.items() if v is not None}

client = GroundClient(host=space_host, port=space_port, timeout=1.5)
packets = client.poll(send_cfg)

# separate telemetry / alerts
telemetry: list[TelemetryPacket] = []
alerts: list[AlertPacket] = []
for p in packets:
    if isinstance(p, TelemetryPacket):
        telemetry.append(p)
    elif isinstance(p, AlertPacket):
        alerts.append(p)

if telemetry and "telemetry_history" not in st.session_state:
    st.session_state["telemetry_history"] = []
if telemetry:
    hist = st.session_state.get("telemetry_history", [])
    hist.extend(telemetry)
    if len(hist) > 60:
        hist[:] = hist[-60:]
    st.session_state["telemetry_history"] = hist

if alerts and "alert_history" not in st.session_state:
    st.session_state["alert_history"] = []
if alerts:
    ah = st.session_state.get("alert_history", [])
    ah.extend(alerts)
    if len(ah) > 100:
        ah[:] = ah[-100:]
    st.session_state["alert_history"] = ah


# ---------------------------------------------------------------------------
# Title bar
# ---------------------------------------------------------------------------
st.title(t("title", lang))
st.caption(t("subtitle", lang))

conn_ok = len(packets) > 0 or len(st.session_state.get("telemetry_history", [])) > 0
st.markdown(
    f"**{'🟢' if conn_ok else '🔴'} {'天基已连接' if conn_ok else '等待天基…'}** | "
    f"tcp://{space_host}:{space_port}"
)


# ---------------------------------------------------------------------------
# Section 1 — Telemetry
# ---------------------------------------------------------------------------
st.header("📡 " + ("遥测信号" if lang == "zh" else "Telemetry Signal"))

hist = st.session_state.get("telemetry_history", [])
if not hist:
    st.info("⏳ " + ("等待天基数据 — 请先启动 python src/space/main.py" if lang == "zh"
                     else "Waiting — start space segment first"))
else:
    n = min(5, len(hist))
    recent = hist[-n:]
    raw_all = np.concatenate([p.raw_values for p in recent])
    t_axis = np.arange(len(raw_all))
    has_scores = all(p.scores is not None for p in recent)

    if has_scores:
        scores_all = np.concatenate([p.scores for p in recent])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("遥测信号" if lang == "zh" else "Telemetry",
                                            "异常分数 (TSPulse)" if lang == "zh" else "Anomaly Score"),
                            row_heights=[0.5, 0.5])
        fig.add_trace(go.Scatter(x=t_axis, y=raw_all, mode="lines",
                                 name="遥测值" if lang == "zh" else "Value",
                                 line=dict(color="#1f77b4", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_axis, y=scores_all, mode="lines",
                                 name="异常分数" if lang == "zh" else "Score",
                                 line=dict(color="#ff7f0e", width=1.5),
                                 fill="tozeroy", fillcolor="rgba(255,127,14,0.12)"), row=2, col=1)
        fig.update_layout(height=420, hovermode="x unified", font=dict(size=12))
        fig.update_xaxes(title_text="时间步" if lang == "zh" else "Time step", row=2, col=1)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_axis, y=raw_all, mode="lines",
                                 name="遥测值" if lang == "zh" else "Value",
                                 line=dict(color="#1f77b4", width=1)))
        fig.update_layout(height=280, hovermode="x unified", font=dict(size=12))

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 2 — Alerts
# ---------------------------------------------------------------------------
st.header("🚨 " + ("告警" if lang == "zh" else "Alerts"))
ah = st.session_state.get("alert_history", [])
if ah:
    for a in reversed(ah[-10:]):
        st.error(f"⚠️ `{a.channel}` | step={a.step} | score={a.score:.4f} | {a.message}")
else:
    st.success("✅ " + ("当前无告警" if lang == "zh" else "No alerts"))


# ---------------------------------------------------------------------------
# Section 3 — Forecaster
# ---------------------------------------------------------------------------
st.header("⚠️ " + ("预警（地基 TTM-R3）" if lang == "zh" else "Early Warning (Ground TTM-R3)"))
if not use_forecast:
    st.caption("预测未启用" if lang == "zh" else "Forecasting disabled")
elif not hist:
    st.info("⏳ " + ("等待数据…" if lang == "zh" else "Waiting for data…"))
else:
    if st.button("▶️ " + ("运行预测" if lang == "zh" else "Run Forecast"), type="primary"):
        latest = hist[-1]
        rw = latest.raw_values
        if len(rw) < 512:
            st.warning("需要 ≥512 点" if lang == "zh" else "Need ≥512 samples")
        else:
            with st.spinner("TTM-R3 " + ("加载模型…" if lang == "zh" else "loading model…")):
                try:
                    forecaster = _load_forecaster()  # lazy-load on first click
                except Exception as e:
                    st.error(f"{('模型加载失败' if lang == 'zh' else 'Model load failed')}: {e}")
                    forecaster = None
            if forecaster is not None:
                with st.spinner("TTM-R3 " + ("预测中…" if lang == "zh" else "forecasting…")):
                    try:
                        ctx, pred = forecaster.forecast(rw[-512:])
                        st.session_state["fc_ctx"] = ctx
                        st.session_state["fc_pred"] = pred
                        st.session_state["fc_done"] = True
                    except Exception as e:
                        st.error(str(e))

    if st.session_state.get("fc_done"):
        ctx = st.session_state.get("fc_ctx")
        pred = st.session_state.get("fc_pred")
        if ctx is not None:
            cx = np.arange(len(ctx))
            px = np.arange(len(ctx), len(ctx) + len(pred))
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=cx, y=ctx, mode="lines",
                                       name="历史" if lang == "zh" else "History",
                                       line=dict(color="#1f77b4", width=1)))
            fig_f.add_trace(go.Scatter(x=px, y=pred, mode="lines",
                                       name="预测" if lang == "zh" else "Forecast",
                                       line=dict(color="#2ca02c", width=2, dash="dash")))
            fig_f.add_vrect(x0=len(ctx), x1=len(ctx) + len(pred),
                            fillcolor="rgba(44,160,44,0.08)",
                            annotation_text="预测区" if lang == "zh" else "Forecast")
            fig_f.update_layout(height=320, hovermode="x unified", font=dict(size=12))
            st.plotly_chart(fig_f, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("预测 σ", f"{float(np.std(pred)):.4f}")
            c2.metric("预测 MSE", f"{float(np.mean(pred**2)):.4f}")


# ---------------------------------------------------------------------------
# Footer & auto-refresh
# ---------------------------------------------------------------------------
st.divider()
st.caption(t("footer", lang))

c1, c2 = st.columns(2)
c1.caption(f"Telemetry packets: {len(hist)} | Alerts: {len(ah)}")
c2.caption(f"Next poll in 1s — auto-refreshing" if lang == "zh" else f"Next poll in 1s — auto-refreshing")

# Auto-poll every ~1 second
time.sleep(1)
st.rerun()


