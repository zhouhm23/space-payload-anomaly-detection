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

Two terminals — start space first, then ground:

```powershell
# Terminal 1 — Space segment (on edge device)
$env:PYTHONPATH = "d:\Office\生产实习\src"; & "d:\Office\生产实习\src\.conda-env\python.exe" -m space.main --host 0.0.0.0

# Terminal 2 — Ground segment (on PC)
$env:PYTHONPATH = "d:\Office\生产实习\src"; & "d:\Office\生产实习\src\.conda-env\python.exe" -m streamlit run src/ground/app.py
```

Open `http://localhost:8501` in browser.

> To connect to a remote edge device, set its IP in sidebar "Connection".

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
│   ├── app.py                 Entry: streamlit run ground/app.py
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

## Data Source

`space/sensor_source.py` simulates a DAQ card with two switchable modes:

1. **DatasetSource** — replays NASA-SMAP/MSL telemetry; returns empty when exhausted
2. **SyntheticSource** — generates continuous signals (multi-sine, sine, square, chirp)

Synthetic mode supports noise injection (missing values, Gaussian noise, jitter);
dataset mode does not need noise settings.

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
