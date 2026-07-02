# Space Payload Health Management System

A lightweight web-based health management demo for space payload telemetry data,
powered by pre-trained time-series foundation models.

## Models

| Module | Model | Role |
|--------|-------|------|
| Anomaly Detection | TSPulse (1M params) | Real-time anomaly scoring on telemetry channels |
| Trend Forecasting | TTM-R3 (5.3M params) | Future value prediction for early warning |

## Quick Start

```bash
# Activate environment
conda activate ./src/.conda-env

# Run the demo
streamlit run src/app/health_monitor.py
```

## Project Structure

```
src/
├── app/                        # Streamlit web application
│   └── health_monitor.py       # Main entry point
├── core/                       # Core logic (model inference, data)
│   ├── anomaly_detection.py    # TSPulse wrapper
│   ├── forecasting.py          # TTM-R3 wrapper
│   └── data_loader.py          # NASA-SMAP/MSL data loading
└── requirements.txt            # Python dependencies
```

## Data

Uses NASA SMAP/MSL telemetry datasets (TSB-UAD version) as a public proxy
for real payload data (which is classified and unavailable).

## Architecture: Space-Ground Collaborative

- **Space segment (simulated):** TSPulse runs lightweight real-time detection
- **Ground segment (simulated):** TTM-R3 performs deeper trend forecasting
- **Collaboration point:** Anomalies detected on-orbit trigger ground-side prediction

## License

Apache-2.0
