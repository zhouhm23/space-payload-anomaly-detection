"""Anomaly detection module using TSPulse (zero-shot reconstruction-based).

This module wraps the TSPulse pre-trained model to provide real-time anomaly
scoring on telemetry channels. It is designed to simulate the "space segment"
(lightweight on-orbit inference) in the space-ground collaborative architecture.
"""

import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction

# Model constants
DEFAULT_MODEL = "ibm-granite/granite-timeseries-tspulse-r1"
CONTEXT_LENGTH = 512


class AnomalyDetector:
    """TSPulse-based anomaly detector for single-channel telemetry.

    Args:
        device: "cuda" or "cpu"
        model_path: HuggingFace model name or local directory path.
                   If None, uses the default online model.
                   For fine-tuned models, pass the local checkpoint directory.
        model_revision: HuggingFace revision (ignored if model_path is local)

    Usage:
        detector = AnomalyDetector(device="cuda")
        scores = detector.detect(values, train_values_for_scaler)
    """

    def __init__(self, device="cuda", model_path=None, model_revision="main"):
        self.device = device
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        path = model_path or DEFAULT_MODEL
        load_kwargs = {}
        # If path is a local directory, don't pass revision
        if path and os.path.isdir(path):
            load_kwargs = {}
        else:
            load_kwargs = {"revision": model_revision}

        self.model = TSPulseForReconstruction.from_pretrained(path, **load_kwargs)
        self.model = self.model.to(device).float().eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.model_source = path

    def detect(self, values, train_values_for_scaler=None):
        """Run anomaly detection on a 1-D telemetry array.

        Computes per-point anomaly scores by comparing the input against
        TSPulse's time-domain and frequency-domain reconstructions.

        Args:
            values: np.ndarray [T] float32 — telemetry values to score
            train_values_for_scaler: np.ndarray or None — training data for StandardScaler

        Returns:
            scores: np.ndarray [T] float32 — anomaly scores (higher = more anomalous),
                    normalized to [0, 1] via MinMaxScaler.
        """
        # Standardize
        if train_values_for_scaler is not None:
            scaler = StandardScaler().fit(train_values_for_scaler.reshape(-1, 1))
        else:
            scaler = StandardScaler().fit(values.reshape(-1, 1))
        scaled = scaler.transform(values.reshape(-1, 1)).flatten().astype(np.float32)

        orig_len = len(scaled)
        T = len(scaled)

        # Ensure enough points: tile if shorter than one window
        if T < CONTEXT_LENGTH:
            repeats = (CONTEXT_LENGTH // T) + 1
            scaled = np.tile(scaled, repeats)
            T = len(scaled)

        # Process in non-overlapping windows of CONTEXT_LENGTH
        n_windows = T // CONTEXT_LENGTH
        all_scores = []

        for w in range(n_windows):
            chunk = scaled[w * CONTEXT_LENGTH:(w + 1) * CONTEXT_LENGTH]
            x = torch.tensor(chunk.reshape(1, -1, 1), dtype=torch.float32,
                             device=self.device)
            with torch.no_grad():
                out = self.model(x)

            # Time-domain reconstruction error (per-point MSE)
            recon = out["reconstruction_outputs"].squeeze(0).squeeze(-1)
            recon = recon.cpu().numpy()
            time_mse = (chunk - recon) ** 2

            # Frequency-domain reconstruction error (per-point MSE)
            fft_recon = out["reconstructed_ts_from_fft"].squeeze(0).squeeze(-1)
            fft_recon = fft_recon.cpu().numpy()
            fft_mse = (chunk - fft_recon) ** 2

            # Combine: take element-wise max of time and frequency scores
            win_scores = np.maximum(time_mse, fft_mse).astype(np.float32)
            all_scores.append(win_scores)

        if not all_scores:
            scores = np.zeros(orig_len, dtype=np.float32)
        else:
            scores = np.concatenate(all_scores)[:orig_len]

        # Normalize to [0, 1]
        mx = float(np.max(scores))
        if mx > 0:
            scores = scores / mx

        return scores
