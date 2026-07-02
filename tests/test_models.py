"""Integration tests for anomaly_detection and forecasting modules.

These tests require GPU/CPU model loading and are slower.
Marked with @pytest.mark.slow — run with: pytest -m slow
Skip with: pytest -m "not slow"
"""

import os
import numpy as np
import pytest

# Set HF cache before importing model modules
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "granite-tsfm", ".hf_cache"),
)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from core.data_loader import list_channels, load_channel, load_train


@pytest.fixture(scope="module")
def sample_channel():
    """Load one MSL channel for testing."""
    channels = list_channels("NASA-MSL")
    name, train_path, test_path = channels[0]
    test_ts, test_labels = load_channel(test_path, train_path)
    train_ts = load_train(train_path) if train_path else None
    return {
        "name": name,
        "test_ts": test_ts,
        "test_labels": test_labels,
        "train_ts": train_ts,
    }


# ---------------------------------------------------------------------------
# Anomaly detection tests
# ---------------------------------------------------------------------------
class TestAnomalyDetector:
    """Tests for AnomalyDetector (TSPulse)."""

    @pytest.mark.slow
    def test_detector_loads(self):
        from core.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector(device="cpu")
        assert detector.n_params > 0
        assert detector.n_params < 5e6, "TSPulse should be < 5M params"

    @pytest.mark.slow
    def test_detect_returns_scores(self, sample_channel):
        from core.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector(device="cpu")
        ts = sample_channel["test_ts"][:1024].astype(np.float32)
        train = sample_channel["train_ts"]
        scores = detector.detect(ts, train)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(ts), "Scores length must match input length"
        assert not np.isnan(scores).any(), "Scores should not contain NaN"

    @pytest.mark.slow
    def test_detect_short_input(self, sample_channel):
        """Detector should handle input shorter than CONTEXT_LENGTH."""
        from core.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector(device="cpu")
        ts = sample_channel["test_ts"][:256].astype(np.float32)  # < 512
        train = sample_channel["train_ts"]
        scores = detector.detect(ts, train)
        assert len(scores) == len(ts), "Scores should match input even when short"

    @pytest.mark.slow
    def test_scores_nonnegative(self, sample_channel):
        """Anomaly scores should be non-negative."""
        from core.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector(device="cpu")
        ts = sample_channel["test_ts"][:1024].astype(np.float32)
        train = sample_channel["train_ts"]
        scores = detector.detect(ts, train)
        assert (scores >= 0).all(), "Anomaly scores should be non-negative"


# ---------------------------------------------------------------------------
# Forecasting tests
# ---------------------------------------------------------------------------
class TestTrendForecaster:
    """Tests for TrendForecaster (TTM-R3)."""

    @pytest.mark.slow
    def test_forecaster_loads(self):
        from core.forecasting import TrendForecaster

        forecaster = TrendForecaster(device="cpu")
        assert forecaster.n_params > 0

    @pytest.mark.slow
    def test_forecast_returns_correct_lengths(self, sample_channel):
        from core.forecasting import TrendForecaster, CONTEXT_LENGTH, PREDICTION_LENGTH

        forecaster = TrendForecaster(device="cpu")
        ts = sample_channel["test_ts"][:1024].astype(np.float32)
        train = sample_channel["train_ts"]
        context, prediction = forecaster.forecast(ts, train)

        assert len(context) == CONTEXT_LENGTH, f"Context should be {CONTEXT_LENGTH} steps"
        assert len(prediction) == PREDICTION_LENGTH, f"Prediction should be {PREDICTION_LENGTH} steps"
        assert not np.isnan(context).any()
        assert not np.isnan(prediction).any()

    @pytest.mark.slow
    def test_forecast_short_input_padded(self, sample_channel):
        """Forecaster should pad input shorter than CONTEXT_LENGTH."""
        from core.forecasting import TrendForecaster, CONTEXT_LENGTH

        forecaster = TrendForecaster(device="cpu")
        ts = sample_channel["test_ts"][:100].astype(np.float32)  # << 512
        train = sample_channel["train_ts"]
        context, prediction = forecaster.forecast(ts, train)
        assert len(context) == CONTEXT_LENGTH
        assert len(prediction) > 0
