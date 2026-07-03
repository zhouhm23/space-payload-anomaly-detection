"""Bilingual (Chinese/English) text dictionary for the health monitor demo.

Usage:
    from core.i18n import t, LANGS
    text = t("title", lang)  # lang = "zh" or "en"
"""

LANGS = {"zh": "中文", "en": "English"}

# All UI strings keyed by identifier.
# Add new keys here with both languages.
_STRINGS = {
    # --- Sidebar ---
    "config": {"zh": "⚙️ 配置", "en": "⚙️ Configuration"},
    "dataset": {"zh": "遥测数据集", "en": "Telemetry Dataset"},
    "dataset_help": {
        "zh": "NASA 航天器遥测数据（真实载荷数据涉密，以此公开数据替代）",
        "en": "NASA spacecraft telemetry (proxy for classified payload data)",
    },
    "channel": {"zh": "通道", "en": "Channel"},
    "channel_help_fmt": {"zh": "共 {} 个通道可选", "en": "{} available"},
    "window_size": {"zh": "显示窗口大小", "en": "Display window size"},
    "lang_label": {"zh": "界面语言", "en": "UI Language"},
    "model_path_label": {"zh": "模型路径（留空=在线下载）", "en": "Model path (blank=online)"},
    "model_path_help": {
        "zh": "微调后填本地权重目录路径",
        "en": "Fill local checkpoint dir after fine-tuning",
    },
    "loading_models": {"zh": "加载模型中...", "en": "Loading models..."},
    "tspulse_loaded": {"zh": "TSPulse 已加载（{}M 参数）[天基]", "en": "TSPulse loaded ({}M params) [Space]"},
    "ttm_loaded": {"zh": "TTM-R3 已加载（{}M 参数）[地基]", "en": "TTM-R3 loaded ({}M params) [Ground]"},
    "load_failed": {"zh": "{} 加载失败：{}", "en": "{} load failed: {}"},
    "architecture": {"zh": "架构", "en": "Architecture"},
    "col_layer": {"zh": "层级", "en": "Layer"},
    "col_model": {"zh": "模型", "en": "Model"},
    "col_role": {"zh": "职能", "en": "Role"},
    "space_seg": {"zh": "🛰️ 天基", "en": "🛰️ Space"},
    "ground_seg": {"zh": "🌍 地基", "en": "🌍 Ground"},
    "role_detect": {"zh": "实时检测", "en": "Real-time detection"},
    "role_forecast": {"zh": "趋势预测", "en": "Trend forecasting"},

    # --- Title ---
    "title": {
        "zh": "🛰️ 空间有效载荷健康管理系统",
        "en": "🛰️ Space Payload Health Management System",
    },
    "subtitle": {
        "zh": "天地协同健康管理 · 异常检测 + 趋势预测 Demo",
        "en": "Space-Ground Collaborative Health Management · Anomaly Detection + Trend Forecasting Demo",
    },

    # --- Section: Telemetry ---
    "telemetry": {"zh": "📡 遥测信号", "en": "📡 Telemetry Signal"},
    "channel_info_fmt": {
        "zh": "通道 **{}** | 数据集 {} | 窗口 {} 点",
        "en": "Channel **{}** | {} | Window: {} points",
    },
    "legend_telemetry": {"zh": "遥测值", "en": "Telemetry"},
    "legend_anomaly_gt": {"zh": "异常（真值标注）", "en": "Anomaly (ground truth)"},
    "xaxis_time": {"zh": "时间步", "en": "Time step"},
    "yaxis_value": {"zh": "数值", "en": "Value"},

    # --- Section: Anomaly Detection ---
    "detection_title": {
        "zh": "🔍 异常检测 — 天基段（TSPulse）",
        "en": "🔍 Anomaly Detection — Space Segment (TSPulse)",
    },
    "detection_desc": {
        "zh": "模拟在轨实时推理，使用 TSPulse（1M 参数，CPU 可运行）",
        "en": "Simulating on-orbit real-time inference with TSPulse (1M params, CPU-capable)",
    },
    "run_detection": {"zh": "▶️ 运行检测", "en": "▶️ Run Detection"},
    "model_label": {"zh": "模型", "en": "Model"},
    "device_label": {"zh": "设备", "en": "Device"},
    "detection_time_label": {"zh": "耗时", "en": "Time"},
    "detection_running": {"zh": "正在运行 TSPulse 异常检测...", "en": "Running TSPulse anomaly detection..."},
    "detection_done": {"zh": "✅ 检测完成，耗时 {:.2f}s", "en": "✅ Detection completed in {:.2f}s"},
    "subplot_telemetry": {"zh": "遥测信号与异常区域", "en": "Telemetry with anomaly regions"},
    "subplot_score": {"zh": "异常分数（TSPulse）", "en": "Anomaly score (TSPulse)"},
    "legend_score": {"zh": "异常分数", "en": "Anomaly score"},
    "detector_not_loaded": {"zh": "异常检测器未加载，请检查模型可用性。", "en": "Anomaly detector not loaded. Check model availability."},

    # --- Section: Early Warning (forecast + detection cascade) ---
    "warning_title": {
        "zh": "⚠️ 预警 — 天→地协同（预测→检测级联）",
        "en": "⚠️ Early Warning — Space-Ground Cascade (Forecast→Detect)",
    },
    "warning_desc": {
        "zh": "地基 TTM-R3 预测未来 96 步 → 天基 TSPulse 检测预测结果 → 若含异常则提前报警。预警提前量 = 96 步。",
        "en": "Ground TTM-R3 forecasts next 96 steps → Space TSPulse detects on forecast → alert if anomalous. Lead time = 96 steps.",
    },
    "run_warning": {"zh": "▶️ 运行预警", "en": "▶️ Run Early Warning"},
    "warning_running": {"zh": "TTM-R3 预测中 → TSPulse 检测预测结果中...", "en": "TTM-R3 forecasting → TSPulse detecting on forecast..."},
    "warning_done": {"zh": "✅ 预警分析完成，耗时 {:.1f}s", "en": "✅ Early warning analysis done in {:.1f}s"},
    "warning_alert": {
        "zh": "🚨 **预警：预测窗口第 {lead} 步检测到异常峰值，建议关注！**（异常分数最高 {score:.3f}）",
        "en": "🚨 **WARNING: Anomaly peak detected at step {lead} in forecast window!** (max score {score:.3f})",
    },
    "warning_clear": {
        "zh": "✅ 预测窗口内未检测到显著异常，遥测趋势正常。",
        "en": "✅ No significant anomaly detected in forecast window. Trend appears normal.",
    },
    "subplot_warning_forecast": {"zh": "预测曲线 + 异常分数", "en": "Forecast curve + anomaly score"},
    "subplot_warning_bar": {"zh": "各步异常分数", "en": "Per-step anomaly score"},
    "warning_lead_time": {"zh": "预警提前量：{} 步", "en": "Warning lead time: {} steps"},

    # --- Section: Forecasting ---
    "forecast_title": {
        "zh": "📈 趋势预测 — 地基段（TTM-R3）",
        "en": "📈 Trend Forecasting — Ground Segment (TTM-R3)",
    },
    "forecast_desc": {
        "zh": "模拟地面深度分析，使用 TTM-R3（5.3M 参数，GIFT-Eval 第3名）",
        "en": "Simulating ground-side deep analysis with TTM-R3 (5.3M params, GIFT-Eval #3)",
    },
    "run_forecast": {"zh": "▶️ 运行预测", "en": "▶️ Run Forecast"},
    "horizon_label": {"zh": "预测步长", "en": "Forecast horizon"},
    "steps_unit": {"zh": "步", "en": "steps"},
    "forecast_running": {"zh": "正在运行 TTM-R3 趋势预测...", "en": "Running TTM-R3 forecasting..."},
    "forecast_done": {"zh": "✅ 预测完成，耗时 {:.2f}s", "en": "✅ Forecast completed in {:.2f}s"},
    "legend_history": {"zh": "历史窗口（标准化）", "en": "History (standardized)"},
    "legend_forecast": {"zh": "预测（TTM-R3）", "en": "Forecast (TTM-R3)"},
    "forecast_zone": {"zh": "预测区", "en": "Forecast zone"},
    "forecast_chart_title_fmt": {
        "zh": "最近 {} 步（历史）→ 未来 {} 步（预测）",
        "en": "Last {} steps (history) → next {} steps (forecast)",
    },
    "yaxis_std": {"zh": "标准化值", "en": "Standardized value"},
    "forecast_mse_info": {
        "zh": "预测 MSE（标准化）：{:.4f} — 误差偏高可能预示前方异常趋势",
        "en": "Forecast MSE (standardized): {:.4f} — elevated error may indicate anomalous trend ahead",
    },
    "forecaster_not_loaded": {"zh": "预测器未加载，请检查模型可用性。", "en": "Forecaster not loaded. Check model availability."},

    # --- Section: Sensor Preprocessing ---
    "preproc_title": {
        "zh": "🔧 传感器数据预处理",
        "en": "🔧 Sensor Data Preprocessing",
    },
    "preproc_desc": {
        "zh": "模拟真实传感器链路：注入噪声/缺失/饱和 → 缺失值插补 → 去噪滤波 → 归一化",
        "en": "Simulate real sensor chain: inject noise/missing/saturation → impute → denoise → normalize",
    },
    "preproc_enable": {"zh": "启用噪声模拟", "en": "Enable noise simulation"},
    "preproc_missing": {"zh": "缺失率", "en": "Missing rate"},
    "preproc_noise": {"zh": "噪声标准差", "en": "Noise std"},
    "preproc_jitter": {"zh": "采样抖动", "en": "Sampling jitter"},
    "preproc_clip": {"zh": "传感器饱和范围", "en": "Sensor saturation range"},
    "preproc_filter": {"zh": "去噪滤波器", "en": "Denoise filter"},
    "preproc_raw": {"zh": "原始传感器（含噪声）", "en": "Raw sensor (with noise)"},
    "preproc_cleaned": {"zh": "预处理后", "en": "After preprocessing"},
    "preproc_stats": {
        "zh": "缺失点：{} | 噪声注入：{} | 滤波方法：{}",
        "en": "Missing: {} | Noise: {} | Filter: {}",
    },

    # --- Footer ---
    "footer": {
        "zh": "模型：TSPulse (ICLR 2026) + TTM-R3 (GIFT-Eval #3) | 数据：NASA SMAP/MSL (Hundman et al. 2018) | 框架：IBM Granite TSFM",
        "en": "Models: TSPulse (ICLR 2026) + TTM-R3 (GIFT-Eval #3) | Data: NASA SMAP/MSL (Hundman et al. 2018) | Framework: IBM Granite TSFM",
    },
}


def t(key, lang="zh", **kwargs):
    """Get a translated string by key.

    Args:
        key: string identifier in _STRINGS
        lang: "zh" or "en"
        **kwargs: format arguments (e.g. t("tspulse_loaded", "zh", n=1.1))

    Returns:
        str: translated and formatted string
    """
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang, entry.get("en", key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text
