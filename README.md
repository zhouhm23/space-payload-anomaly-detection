
# Space Payload Health Management System

Space-ground collaborative health management for space payload telemetry.
The space segment runs on-orbit anomaly detection; the ground segment
displays telemetry and runs trend forecasting for early warning.

## Architecture

```
┌─────────── Space Segment (on-orbit) ───────────┐
│  Independent process — deployable to edge HW    │
│                                                 │
│  SensorSource → Preprocess → TSPulse Detect     │
│  (DAQ card)    (impute+norm)  (anomaly score)   │
│                                                 │
│        TCP Server (0.0.0.0:9876)                │
│        Receives config / sends telemetry+alerts │
└─────────────────────────────────────────────────┘
                    │ TCP (WiFi)
                    ▼
┌─────────── Ground Segment (ground station) ────┐
│  Independent process — runs on PC               │
│                                                 │
│  TCP Client → Streamlit UI → TTM-R3 Forecast    │
│  (polls space)  (waveform+alerts)  (96-step)    │
│                                                 │
│  Sidebar controls space: source/channel/noise   │
└─────────────────────────────────────────────────┘
```

## Quick Start

### 1. Space Segment (on-orbit node)

```bash
python -m space.main
```

Edit `DAQ_CONFIG` in `space/main.py` to configure the DAQ card channels and
sensors.  Restart to apply changes.

```python
DAQ_CONFIG = {
    "sample_rate": 100.0,     # Hz
    "window_size": 512,
    "channels": [
        {"id": 0, "source_id": "file:NASA-MSL/C-1", "loop": True, "enabled": True},
        {"id": 1, "source_id": "virtual:sine",       "loop": False, "signal_freq_hz": 2.0, "enabled": False},
    ],
}
```

See available sources: `python -c "from space.sensor_source import list_all_sources; [print(s['id']) for s in list_all_sources()]"`

### 2. Ground Segment (ECharts HTML frontend)

```bash
python ground/server.py
```

Open `http://localhost:8501`.  The HTML frontend provides:

- **Device tree** — create/edit sensors and racks, drag-and-drop, save config
- **Telemetry charts** — ECharts real-time waveforms + anomaly scores
- **Alerts & early warnings** — TSPulse detection + TTM-R3 forecast cascade

### 3. (Legacy) Streamlit frontend

```bash
streamlit run ground/app.py   # alternative ground UI
```

## Directory Structure

```
src/
├── space/                     Space segment (self-contained)
│   ├── main.py                Entry CLI: python -m space.main
│   ├── preprocessing.py       Impute + normalize (no filtering)
│   ├── sensor_source.py       Simulated DAQ (dataset replay / synthetic)
│   ├── comm.py                TCP server
│   ├── data_loader.py         NASA-SMAP/MSL data loading
│   ├── anomaly_detection.py   TSPulse anomaly detection (direct MSE, per-point)
│   └── tests/
├── ground/                    Ground segment (self-contained)
│   ├── server.py              FastAPI: serves HTML + TCP bridge
│   ├── app.py                 Legacy Streamlit entry
│   ├── 空间站有效载荷…html     ECharts frontend
│   ├── settings.json          Persisted settings
│   ├── comm.py                TCP client (polls space + sends config)
│   ├── forecasting.py         TTM-R3 trend forecasting
│   ├── i18n.py                Bilingual (zh/en) text
│   ├── data_loader.py         NASA data loading
│   ├── anomaly_detection.py   TSPulse detection (reused on ground)
│   └── tests/
├── tests/
│   └── test_e2e.py            End-to-end integration test
├── .conda-env/                Shared conda environment
├── .hf_cache/                 Model weight cache (git-ignored)
├── pytest.ini
├── requirements.txt
└── README.md
```

## Data Sources

`space/sensor_source.py` simulates a DAQ card with a unified interface:

| Type                          | `source_id`         | Behavior                                                                    |
| ----------------------------- | --------------------- | --------------------------------------------------------------------------- |
| **FileSource**          | `file:NASA-MSL/C-1` | Replays NASA-SMAP/MSL telemetry;`loop=True` rewinds on exhaustion         |
| **VirtualSensorSource** | `virtual:sine`      | Continuous synthetic signals;`signal_freq_hz` controls waveform frequency |

All sources share the `SensorSource.read(n)` interface — the space segment
does not distinguish real from virtual sensors.

Virtual sensors support noise injection (missing values, Gaussian noise, jitter).

## Preprocessing

`space/preprocessing.py` only converts format — does not alter signal features:

- Linear interpolation for NaN (missing-value imputation)
- StandardScaler normalization

**No filtering/denoising** — preserves anomaly characteristics.

## Anomaly Detection

`space/anomaly_detection.py` uses TSPulse for zero-shot reconstruction-based
anomaly scoring:

- Runs the model forward to get `reconstruction_outputs` (time-domain) and
  `reconstructed_ts_from_fft` (frequency-domain)
- Computes per-point MSE for both, takes element-wise max
- Normalizes scores to [0, 1]
- Returns one score per input point (not aggregated)

> Previously used `TimeSeriesAnomalyDetectionPipeline` with
> `aggregation_length=64`, which collapsed 512 points into a single score
> and produced all-zero output after MinMaxScaler. Switched to direct
> model-forward MSE for per-point resolution.

## Communication Protocol

Space TCP server (default `0.0.0.0:9876`):

- Ground sends one JSON config line on each connection (source/channel/noise/rate)
- Space reconfigures dynamically, then returns buffered telemetry + alerts
- One JSON object per line, terminated by `END`

## Testing

### Unit tests (fast, no model loading)

```powershell
cd src
.\..\.conda-env\python.exe -m pytest space/tests ground/tests -q
```

### End-to-end test (spawns space subprocess, 6 tests, ~40 s)

```powershell
cd src
$env:HF_HUB_OFFLINE = "1"
$env:HF_HOME = "$PWD\.hf_cache"
.\..\.conda-env\python.exe -m pytest tests/test_e2e.py -v -s
```

> Requires `HF_HUB_OFFLINE=1` and `HF_HOME` pointing to `.hf_cache` so the
> space subprocess and test process can load TSPulse / TTM-R3 weights
> from local cache without network access.

The e2e test uses **golden-value verification** — each test checks against
pre-computed ground truth, not just "data exists":

| Test   | What it verifies                                                               |
| ------ | ------------------------------------------------------------------------------ |
| test_1 | MSL C-1 telemetry**exactly matches** a 512-pt window in the dataset file |
| test_2 | Detection scores are non-zero, non-negative, majority nonzero                  |
| test_3 | After MSL→SMAP switch, telemetry**exactly matches** SMAP E-1 dataset    |
| test_4 | Config change takes effect within 5 s (catches pace-loop blocking)             |
| test_5 | TTM-R3 forecast produces 96 steps with no NaN                                  |
| test_6 | Space shutdown handled gracefully                                              |

## Data

NASA SMAP/MSL telemetry datasets (TSB-UAD version) as a public proxy for
classified real payload data.

## License

Apache-2.0
