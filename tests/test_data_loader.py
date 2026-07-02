"""Unit tests for data_loader — no model dependency, fast."""

import numpy as np
from core.data_loader import list_channels, load_channel, load_train


class TestListChannels:
    """Tests for list_channels()."""

    def test_msl_has_channels(self):
        channels = list_channels("NASA-MSL")
        assert len(channels) > 0, "MSL should have at least one channel"

    def test_smap_has_channels(self):
        channels = list_channels("NASA-SMAP")
        assert len(channels) > 0, "SMAP should have at least one channel"

    def test_channel_tuple_format(self):
        channels = list_channels("NASA-MSL")
        ch = channels[0]
        assert len(ch) == 3, "Each channel should be (name, train_path, test_path)"
        name, train_path, test_path = ch
        assert isinstance(name, str)
        assert test_path is not None
        assert test_path.endswith(".test.out")

    def test_nonexistent_dataset(self):
        channels = list_channels("NONEXISTENT")
        assert channels == [], "Nonexistent dataset should return empty list"


class TestLoadChannel:
    """Tests for load_channel()."""

    def test_load_returns_correct_types(self):
        channels = list_channels("NASA-MSL")
        name, train_path, test_path = channels[0]
        ts, labels = load_channel(test_path, train_path)
        assert isinstance(ts, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert ts.dtype == np.float32
        assert labels.dtype == int or labels.dtype == np.int64 or labels.dtype == np.int32

    def test_load_shapes_match(self):
        channels = list_channels("NASA-MSL")
        name, train_path, test_path = channels[0]
        ts, labels = load_channel(test_path, train_path)
        assert len(ts) == len(labels), "Time series and labels must have same length"

    def test_labels_are_binary(self):
        channels = list_channels("NASA-MSL")
        name, train_path, test_path = channels[0]
        _, labels = load_channel(test_path, train_path)
        unique = set(np.unique(labels))
        assert unique.issubset({0, 1}), f"Labels should be binary, got {unique}"

    def test_at_least_one_channel_has_anomalies(self):
        """At least one MSL channel should contain anomaly labels."""
        channels = list_channels("NASA-MSL")
        found_anomaly = False
        for name, train_path, test_path in channels:
            _, labels = load_channel(test_path, train_path)
            if labels.sum() > 0:
                found_anomaly = True
                break
        assert found_anomaly, "At least one MSL channel should have anomalies"


class TestLoadTrain:
    """Tests for load_train()."""

    def test_load_train_returns_array(self):
        channels = list_channels("NASA-MSL")
        name, train_path, test_path = channels[0]
        if train_path:
            ts = load_train(train_path)
            assert isinstance(ts, np.ndarray)
            assert len(ts) > 0

    def test_load_train_none_path(self):
        result = load_train(None)
        assert result is None
