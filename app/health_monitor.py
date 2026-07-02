"""Space Payload Health Management — Streamlit Demo (Bilingual).

This demo integrates:
  1. Telemetry visualization (NASA-SMAP/MSL waveforms with anomaly labels)
  2. Real-time anomaly detection (TSPulse, simulating on-orbit inference)
  3. Future trend forecasting (TTM-R3, simulating ground-side deep analysis)

Run:
    streamlit run src/app/health_monitor.py
"""

import os
import sys
import time
import numpy as np

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Resolve paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_PROJ = os.path.dirname(_SRC)

# HF cache isolation (D drive, not C drive)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault(
    "HF_HOME", os.path.join(_PROJ, "baselines", "granite-tsfm", ".hf_cache")
)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Add src to path for core imports
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.data_loader import list_channels, load_channel, load_train
from core.anomaly_detection import AnomalyDetector
from core.forecasting import TrendForecaster
from core.i18n import t, LANGS


# ---------------------------------------------------------------------------
# Page config & language state
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Space Payload Health Monitor",
    page_icon="🛰️",
    layout="wide",
)

if "lang" not in st.session_state:
    st.session_state["lang"] = "zh"
lang = st.session_state["lang"]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header(t("config", lang))

    # Language switcher
    lang_choice = st.selectbox(
        t("lang_label", lang),
        list(LANGS.keys()),
        format_func=lambda k: LANGS[k],
        index=list(LANGS.keys()).index(lang),
    )
    if lang_choice != lang:
        st.session_state["lang"] = lang_choice
        st.rerun()

    # Dataset selection
    dataset = st.selectbox(
        t("dataset", lang),
        ["NASA-MSL", "NASA-SMAP"],
        help=t("dataset_help", lang),
    )

    # List channels
    channels = list_channels(dataset)
    channel_names = [c[0] for c in channels]
    default_idx = 0
    if "C-1" in channel_names:
        default_idx = channel_names.index("C-1")
    selected_channel = st.selectbox(
        f"{t('channel', lang)} ({t('channel_help_fmt', lang).format(len(channel_names))})",
        channel_names,
        index=default_idx,
    )

    # Window size
    window_size = st.slider(
        t("window_size", lang), min_value=512, max_value=4096, value=2048, step=256
    )

    # Model paths — folded under expander (advanced settings)
    with st.expander(t("model_path_label", lang), expanded=False):
        tspulse_path_input = st.text_input(
            "TSPulse", value="", placeholder=t("model_path_help", lang), key="tspulse_path"
        )
        ttm_path_input = st.text_input(
            "TTM-R3", value="", placeholder=t("model_path_help", lang), key="ttm_path"
        )


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
# Model loading (cached — downloads once, reuses across restarts)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector(device="cuda", model_path=""):
    path = model_path.strip() if model_path else None
    return AnomalyDetector(device=device, model_path=path)


@st.cache_resource
def load_forecaster(device="cuda", model_path=""):
    path = model_path.strip() if model_path else None
    return TrendForecaster(device=device, model_path=path)


device = "cuda"
try:
    import torch
    if not torch.cuda.is_available():
        device = "cpu"
except ImportError:
    device = "cpu"

# Load models (cached: downloads only on first run, instant thereafter)
with st.spinner("⏳ Loading models (first time may take ~15s to download weights)..."):
    tspulse_p = tspulse_path_input.strip() if tspulse_path_input else ""
    ttm_p = ttm_path_input.strip() if ttm_path_input else ""
    detector = load_detector(device, tspulse_p)
    forecaster = load_forecaster(device, ttm_p)

with st.sidebar:
    st.divider()
    st.success(t("tspulse_loaded", lang).format(detector.n_params / 1e6))
    st.success(t("ttm_loaded", lang).format(forecaster.n_params / 1e6))

    st.divider()
    st.markdown(f"### {t('architecture', lang)}")
    st.markdown(
        f"| {t('col_layer',lang)} | {t('col_model',lang)} | {t('col_role',lang)} |\n"
        f"|---|---|---|\n"
        f"| {t('space_seg',lang)} | TSPulse (1M) | {t('role_detect',lang)} |\n"
        f"| {t('ground_seg',lang)} | TTM-R3 (5M) | {t('role_forecast',lang)} |"
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ch_info = next(c for c in channels if c[0] == selected_channel)
ch_name, train_path, test_path = ch_info

with st.spinner(f"Loading channel {ch_name}..."):
    test_ts, test_labels = load_channel(test_path, train_path)
    train_ts = load_train(train_path) if train_path else None

# Slice to display window (prefer a window containing anomalies)
if len(test_ts) > window_size:
    anomaly_indices = np.where(test_labels == 1)[0]
    if len(anomaly_indices) > 0:
        center = anomaly_indices[len(anomaly_indices) // 2]
        start = max(0, center - window_size // 2)
        start = min(start, len(test_ts) - window_size)
    else:
        start = 0
    display_ts = test_ts[start : start + window_size]
    display_labels = test_labels[start : start + window_size]
else:
    display_ts = test_ts[:window_size]
    display_labels = test_labels[:window_size]
    start = 0

time_axis = np.arange(len(display_ts))
anomaly_mask = display_labels == 1

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title(t("title", lang))
st.caption(t("subtitle", lang))

# ---------------------------------------------------------------------------
# Section 1: Telemetry waveform
# ---------------------------------------------------------------------------
st.header(t("telemetry", lang))
st.caption(
    t("channel_info_fmt", lang).format(ch_name, dataset, len(display_ts))
)

fig_wave = go.Figure()
fig_wave.add_trace(
    go.Scatter(
        x=time_axis,
        y=display_ts,
        mode="lines",
        name=t("legend_telemetry", lang),
        line=dict(color="#1f77b4", width=1),
    )
)
if anomaly_mask.any():
    fig_wave.add_trace(
        go.Scatter(
            x=time_axis[anomaly_mask],
            y=display_ts[anomaly_mask],
            mode="markers",
            name=t("legend_anomaly_gt", lang),
            marker=dict(color="red", size=4),
        )
    )
fig_wave.update_layout(
    height=300,
    xaxis_title=t("xaxis_time", lang),
    yaxis_title=t("yaxis_value", lang),
    hovermode="x unified",
)
st.plotly_chart(fig_wave, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2: Anomaly Detection (Space Segment)
# ---------------------------------------------------------------------------
st.header(t("detection_title", lang))
st.caption(t("detection_desc", lang))

run_detection = st.button(t("run_detection", lang), type="primary", key="btn_detect")
col_a1, col_a2, col_a3 = st.columns(3)
col_a1.metric(t("model_label", lang), "TSPulse 1.1M")
col_a2.metric(t("device_label", lang), device.upper())

if run_detection or st.session_state.get("detection_done", False):
    with st.spinner(t("detection_running", lang)):
        if run_detection:
            t0 = time.time()
            scaler_data = train_ts if train_ts is not None else display_ts
            scores = detector.detect(display_ts, scaler_data)
            elapsed = time.time() - t0
            st.session_state["detection_scores"] = scores
            st.session_state["detection_done"] = True
            st.session_state["detection_time"] = elapsed

    scores = st.session_state.get("detection_scores")
    if scores is not None:
        elapsed = st.session_state.get("detection_time", 0)
        col_a3.metric(t("detection_time_label", lang), f"{elapsed:.2f}s")
        st.success(t("detection_done", lang).format(elapsed))

        fig_det = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                t("subplot_telemetry", lang),
                t("subplot_score", lang),
            ),
            row_heights=[0.5, 0.5],
        )
        fig_det.add_trace(
            go.Scatter(
                x=time_axis, y=display_ts, mode="lines",
                name=t("legend_telemetry", lang),
                line=dict(color="#1f77b4", width=1),
            ),
            row=1, col=1,
        )
        if anomaly_mask.any():
            fig_det.add_trace(
                go.Scatter(
                    x=time_axis[anomaly_mask], y=display_ts[anomaly_mask],
                    mode="markers", name=t("legend_anomaly_gt", lang),
                    marker=dict(color="red", size=4),
                ),
                row=1, col=1,
            )
        fig_det.add_trace(
            go.Scatter(
                x=time_axis, y=scores, mode="lines",
                name=t("legend_score", lang),
                line=dict(color="#ff7f0e", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,127,14,0.1)",
            ),
            row=2, col=1,
        )
        fig_det.update_layout(
            height=450, hovermode="x unified",
            font=dict(size=12),
        )
        fig_det.update_xaxes(title_text=t("xaxis_time", lang), row=2, col=1)
        st.plotly_chart(fig_det, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Early Warning (Forecast → Detect Cascade)
# ---------------------------------------------------------------------------
st.header(t("warning_title", lang))
st.caption(t("warning_desc", lang))

run_warning = st.button(t("run_warning", lang), type="primary", key="btn_warn")
col_w1, col_w2, col_w3 = st.columns(3)
col_w1.metric(t("model_label", lang), "TSPulse+TTM")
col_w2.metric(t("horizon_label", lang), f"96 {t('steps_unit', lang)}")

if run_warning or st.session_state.get("warning_done", False):
    with st.spinner(t("warning_running", lang)):
        if run_warning:
            t0 = time.time()
            scaler_data = train_ts if train_ts is not None else display_ts
            ctx_input = display_ts[-512:] if len(display_ts) >= 512 else display_ts
            context_std, prediction_std = forecaster.forecast(ctx_input, scaler_data)
            fcast_scores = detector.detect(prediction_std, prediction_std)
            elapsed = time.time() - t0
            st.session_state["warn_ctx"] = context_std
            st.session_state["warn_pred"] = prediction_std
            st.session_state["warn_scores"] = fcast_scores
            st.session_state["warning_done"] = True
            st.session_state["warning_time"] = elapsed

    prediction = st.session_state.get("warn_pred")
    context = st.session_state.get("warn_ctx")
    fcast_scores = st.session_state.get("warn_scores")
    if prediction is not None and fcast_scores is not None:
        elapsed = st.session_state.get("warning_time", 0)
        col_w3.metric(t("detection_time_label", lang), f"{elapsed:.2f}s")
        st.success(t("warning_done", lang).format(elapsed))

        max_score = float(np.max(fcast_scores))
        max_idx = int(np.argmax(fcast_scores))
        threshold = 0.5 * max_score if max_score > 0.01 else 0.01
        n_alert_steps = int((fcast_scores > threshold).sum())

        if n_alert_steps > 0:
            st.error(t("warning_alert", lang).format(lead=max_idx + 1, score=max_score))
        else:
            st.info(t("warning_clear", lang))

        fig_warn = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                t("subplot_warning_forecast", lang),
                t("subplot_warning_bar", lang),
            ),
            row_heights=[0.55, 0.45],
        )
        ctx_x = np.arange(len(context))
        pred_x = np.arange(len(context), len(context) + len(prediction))
        fig_warn.add_trace(
            go.Scatter(
                x=ctx_x, y=context, mode="lines",
                name=t("legend_history", lang),
                line=dict(color="#1f77b4", width=1),
            ), row=1, col=1,
        )
        fig_warn.add_trace(
            go.Scatter(
                x=pred_x, y=prediction, mode="lines",
                name=t("legend_forecast", lang),
                line=dict(color="#2ca02c", width=2, dash="dash"),
            ), row=1, col=1,
        )
        fig_warn.add_vrect(
            x0=len(context), x1=len(context) + len(prediction),
            fillcolor="rgba(44,160,44,0.08)",
            annotation_text=t("forecast_zone", lang),
            row=1, col=1,
        )
        colors = ["#ff7f0e" if s > threshold else "#1f77b4" for s in fcast_scores]
        fig_warn.add_trace(
            go.Bar(
                x=pred_x, y=fcast_scores,
                name=t("legend_score", lang),
                marker_color=colors,
            ), row=2, col=1,
        )
        fig_warn.add_hline(
            y=threshold, line_dash="dash", line_color="red",
            annotation_text=f"threshold={threshold:.3f}",
            row=2, col=1,
        )
        fig_warn.update_layout(
            height=480, hovermode="x unified",
            font=dict(size=12),
        )
        fig_warn.update_xaxes(title_text=t("xaxis_time", lang), row=2, col=1)
        st.plotly_chart(fig_warn, use_container_width=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(t("warning_lead_time", lang).format(""), f"{max_idx + 1} steps")
        col_m2.metric("Max anomaly score", f"{max_score:.4f}")
        col_m3.metric("Alert steps (>threshold)", f"{n_alert_steps}/{len(fcast_scores)}")

# ---------------------------------------------------------------------------
# Section 4: Trend Forecasting (Ground Segment) — standalone
# ---------------------------------------------------------------------------
st.header(t("forecast_title", lang))
st.caption(t("forecast_desc", lang))

run_forecast = st.button(t("run_forecast", lang), type="primary", key="btn_forecast")
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.metric(t("model_label", lang), "TTM-R3 5.3M")
col_f2.metric(t("horizon_label", lang), f"96 {t('steps_unit', lang)}")

if run_forecast or st.session_state.get("forecast_done", False):
    with st.spinner(t("forecast_running", lang)):
        if run_forecast:
            t0 = time.time()
            scaler_data = train_ts if train_ts is not None else display_ts
            ctx_input = display_ts[-512:] if len(display_ts) >= 512 else display_ts
            context_std, prediction_std = forecaster.forecast(ctx_input, scaler_data)
            elapsed = time.time() - t0
            st.session_state["forecast_ctx"] = context_std
            st.session_state["forecast_pred"] = prediction_std
            st.session_state["forecast_done"] = True
            st.session_state["forecast_time"] = elapsed

    prediction = st.session_state.get("forecast_pred")
    context = st.session_state.get("forecast_ctx")
    if prediction is not None and context is not None:
        elapsed = st.session_state.get("forecast_time", 0)
        col_f3.metric(t("detection_time_label", lang), f"{elapsed:.2f}s")
        st.success(t("forecast_done", lang).format(elapsed))

        fig_fc = go.Figure()
        ctx_x = np.arange(len(context))
        pred_x = np.arange(len(context), len(context) + len(prediction))
        fig_fc.add_trace(
            go.Scatter(
                x=ctx_x, y=context, mode="lines",
                name=t("legend_history", lang),
                line=dict(color="#1f77b4", width=1.5),
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=pred_x, y=prediction, mode="lines",
                name=t("legend_forecast", lang),
                line=dict(color="#2ca02c", width=2, dash="dash"),
            )
        )
        fig_fc.add_vrect(
            x0=len(context), x1=len(context) + len(prediction),
            fillcolor="rgba(44,160,44,0.1)",
            annotation_text=t("forecast_zone", lang),
        )
        fig_fc.update_layout(
            height=350,
            title=t("forecast_chart_title_fmt", lang).format(len(context), len(prediction)),
            xaxis_title=t("xaxis_time", lang),
            yaxis_title=t("yaxis_std", lang),
            hovermode="x unified",
            font=dict(size=12),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        pred_error = float(np.mean(prediction ** 2))
        st.info(t("forecast_mse_info", lang).format(pred_error))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(t("footer", lang))
