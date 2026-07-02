# Data loading utilities for NASA-SMAP/MSL telemetry (TSB-UAD format).

import os
import glob
import numpy as np
import pandas as pd


# Resolve project root relative to this file: src/core/ -> src/ -> project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TSB_ROOT = os.path.join(_PROJ, "datasets", "TSB-UAD-Public")


def list_channels(dataset_name):
    """Return list of (channel_name, train_path, test_path) for a NASA dataset.

    TSB-UAD naming: <channel>.test.out (test), <channel>.train.out (train)
    """
    base = os.path.join(_TSB_ROOT, dataset_name)
    test_files = sorted(glob.glob(os.path.join(base, "*.test.out")))
    channels = []
    for test_path in test_files:
        fname = os.path.basename(test_path)
        ch = fname.replace(".test.out", "")
        train_path = os.path.join(base, f"{ch}.train.out")
        if not os.path.exists(train_path):
            train_path = None
        channels.append((ch, train_path, test_path))
    return channels


def load_channel(test_path, train_path=None):
    """Load a single TSB-UAD .out file.

    Returns:
        ts: np.ndarray [T] float32 — telemetry values
        labels: np.ndarray [T] int — ground-truth anomaly labels (0/1)
    """
    arr = np.loadtxt(test_path, delimiter=",")
    ts = arr[:, 0].astype(np.float32)
    labels = arr[:, 1].astype(int)
    return ts, labels


def load_train(train_path):
    """Load training data for scaler fitting."""
    if train_path and os.path.exists(train_path):
        arr = np.loadtxt(train_path, delimiter=",")
        return arr[:, 0].astype(np.float32)
    return None
