"""RUL prediction plugin — LSTM + Attention (trained on C-MAPSS).

Wraps the trained LSTM+Attention weights (``src/ground/models/rul/
fd00X_lstm_attn.pt``) into a lightweight inference class that follows the
``BaseRULPredictor`` contract.  The architecture definition is duplicated
here (not imported from ``experiments/``) to keep the product library
self-contained — ``experiments/`` is private-research code and must not be
referenced from ``src/`` (see AGENTS.md "library boundary").

The class is lazy-loading: the PyTorch model is only instantiated and
weights loaded on the first ``predict_rul`` call.  This keeps the ground
server startup fast even when the RUL feature is not exercised.

**Normalisation**: callers pass **raw** sensor readings (physical units, e.g.
T24 ≈ 641 K) and the class normalises internally using the Min-Max scaler
fitted on the training set and persisted as ``scaler_fd00X.json`` next to
the checkpoint.  This mirrors the training pipeline
(``experiments/rul/data_loader._normalise_sensors``) so online inference
matches training bit-for-bit.  Pass ``raw=False`` to bypass normalisation
if the input is already normalised (legacy/advanced use).

Usage::

    from phm.algorithm import RULPredictor

    rul = RULPredictor(subset="FD001")          # lazy, no model loaded yet
    remaining = rul.predict_rul(raw_window)      # (30, 14) raw → 87.5 cycles
"""

from __future__ import annotations

import json
import os
import logging

import numpy as np
import torch
import torch.nn as nn

from .base import BaseRULPredictor

__all__ = ["RULPredictor"]

log = logging.getLogger(__name__)

# ── Default paths ────────────────────────────────────────────────────────
# Models live under src/ground/models/rul/ (product-library assets).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "models", "rul"))

# ── Architecture constants (must match experiments/rul/model.py) ─────────
# 14 sensors carry degradation info on C-MAPSS FD001-FD004.
N_SENSORS = 14
DEFAULT_WINDOW_SIZE = 30
DEFAULT_MAX_RUL = 125


# ---------------------------------------------------------------------------
# Architecture (duplicated from experiments/rul/model.py for self-containment)
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM timesteps."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        weights = self.score(lstm_out)            # (B, L, 1)
        weights = torch.softmax(weights, dim=1)   # normalise over L
        context = (lstm_out * weights).sum(dim=1)  # (B, H)
        return context


class _LSTMAttentionRUL(nn.Module):
    """LSTM + Attention regression model for RUL prediction."""

    def __init__(
        self,
        n_sensors: int = N_SENSORS,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = _Attention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)          # (B, L, H)
        context = self.attention(lstm_out)  # (B, H)
        rul = self.head(context)            # (B, 1)
        return rul.squeeze(-1)              # (B,)


# ---------------------------------------------------------------------------
# Public inference wrapper
# ---------------------------------------------------------------------------

class RULPredictor(BaseRULPredictor):
    """LSTM+Attention RUL predictor trained on NASA C-MAPSS.

    Loads a trained checkpoint (``fd00X_lstm_attn.pt``) and exposes a
    simple ``predict_rul(window)`` interface.  The model is lazy-loaded
    on first prediction to keep server startup fast.

    Args:
        subset: C-MAPSS subset — ``"FD001"`` / ``"FD002"`` / ``"FD003"``
                / ``"FD004"``.  Determines which checkpoint to load.
        device: ``"cuda"`` / ``"cpu"`` / ``"auto"``.
        models_dir: override checkpoint directory (defaults to
                    ``src/ground/models/rul/``).
    """

    def __init__(
        self,
        subset: str = "FD001",
        device: str = "auto",
        models_dir: str | None = None,
    ) -> None:
        self.subset = subset.upper()
        if self.subset not in ("FD001", "FD002", "FD003", "FD004"):
            raise ValueError(f"Unsupported subset: {subset}")
        self._models_dir = models_dir or _MODELS_DIR
        self._weights_path = os.path.join(
            self._models_dir, f"{self.subset.lower()}_lstm_attn.pt"
        )
        self._scaler_path = os.path.join(
            self._models_dir, f"scaler_{self.subset.lower()}.json"
        )

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Lazy-loaded state.
        self._model: _LSTMAttentionRUL | None = None
        self._config: dict = {}
        self._scaler: dict | None = None  # {"min":[...],"range":[...]}
        self.n_params: int = 0
        self.model_source: str = self._weights_path

    # ── Lazy loading ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load model weights on first call (idempotent)."""
        if self._model is not None:
            return
        if not os.path.exists(self._weights_path):
            raise FileNotFoundError(
                f"RUL checkpoint not found: {self._weights_path}. "
                f"Run experiments/rul/train_rul.py --subset {self.subset} first."
            )
        ckpt = torch.load(
            self._weights_path, map_location=self._device, weights_only=False
        )
        self._config = ckpt.get("config", {})
        cfg = self._config
        n_sensors = cfg.get("n_sensors", N_SENSORS)
        hidden_dim = cfg.get("hidden_dim", 64)
        n_layers = cfg.get("n_layers", 2)
        dropout = cfg.get("dropout", 0.2)

        model = _LSTMAttentionRUL(
            n_sensors=n_sensors,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self._device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Detect meta-tensor corruption: under rare race conditions (e.g. the
        # TSPulse pipeline's from_pretrained failing mid-download and leaving
        # torch's init_empty_weights context partially active), newly-created
        # modules can end up with meta tensors that have no data.  Catch this
        # here with a clear message rather than letting the subsequent .to()
        # raise an opaque NotImplementedError.
        bad = [n for n, p in model.named_parameters() if p.is_meta]
        if bad:
            raise RuntimeError(
                f"RUL model {self.subset} has {len(bad)} meta-tensor params "
                f"(e.g. {bad[0]}). This indicates torch's module-init state "
                f"was corrupted (often by a concurrent from_pretrained failure). "
                f"Retry; if it persists, load this model before any tsfm_public "
                f"model. Param names: {bad[:5]}"
            )

        self._model = model
        self.n_params = sum(p.numel() for p in model.parameters())

        # Load the Min-Max scaler fitted on the training set.  Without it,
        # predictions on raw sensor values would be garbage (the LSTM was
        # trained on normalised [0,1] inputs).  A missing scaler is a soft
        # error — log a warning so callers know raw=True will misbehave, but
        # don't crash (lets raw=False legacy callers still work).
        if os.path.exists(self._scaler_path):
            with open(self._scaler_path, "r", encoding="utf-8") as f:
                self._scaler = json.load(f)
        else:
            log.warning(
                "RUL scaler not found: %s. Raw-input prediction (raw=True) "
                "will produce incorrect results. Run "
                "experiments/rul/export_scaler.py to generate it.",
                self._scaler_path,
            )

        log.info(
            "RULPredictor loaded %s (%.2fM params, best_epoch=%d, best_rmse=%.3f, scaler=%s)",
            self.subset, self.n_params / 1e6,
            ckpt.get("best_epoch", -1), ckpt.get("best_rmse", -1),
            "loaded" if self._scaler else "MISSING",
        )

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def window_size(self) -> int:
        """Expected input window length (cycles)."""
        if self._model is None:
            return self._config.get("window_size", DEFAULT_WINDOW_SIZE)
        return self._config.get("window_size", DEFAULT_WINDOW_SIZE)

    @property
    def max_rul(self) -> int:
        """RUL cap used during training (predictions are clipped to this)."""
        return self._config.get("max_rul", DEFAULT_MAX_RUL)

    def predict_rul(self, window: np.ndarray, raw: bool = True) -> float:
        """Predict remaining useful life from a sensor window.

        Args:
            window: ``(window_size, n_sensors)`` sensor readings.  By default
                    ``raw=True`` — pass **physical-unit** readings (e.g. T24
                    ≈ 641 K) and the class normalises internally using the
                    persisted Min-Max scaler, exactly as during training.
                    Pass ``raw=False`` if the input is already normalised to
                    [0,1] (legacy/advanced use).
                    Minor shape mismatch (n_sensors correct but length
                    differs) is handled by padding / truncation.
            raw: whether the input is in physical units (True, default) or
                 already normalised (False).

        Returns:
            Predicted RUL in cycles, clipped to ``[0, max_rul]``.
        """
        self._ensure_loaded()
        x = self._prepare_window(window, raw=raw)
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(self._device)
            pred = self._model(x_t)
        rul = float(pred.squeeze().cpu().item())
        return max(0.0, min(rul, float(self.max_rul)))

    def predict_rul_batch(
        self, windows: np.ndarray, raw: bool = True
    ) -> np.ndarray:
        """Batch RUL prediction.

        Args:
            windows: ``(N, window_size, n_sensors)`` array.  See
                     :meth:`predict_rul` for the ``raw`` flag semantics.
            raw: whether inputs are in physical units (True, default) or
                 already normalised (False).

        Returns:
            ``(N,)`` array of predicted RUL values, clipped to
            ``[0, max_rul]``.
        """
        self._ensure_loaded()
        if windows.ndim != 3:
            raise ValueError(
                f"Expected 3-D array (N, window_size, n_sensors), got {windows.shape}"
            )
        prepared = np.stack([self._prepare_window(w, raw=raw) for w in windows])
        with torch.no_grad():
            x_t = torch.from_numpy(prepared).to(self._device)
            preds = self._model(x_t)
        rul = preds.squeeze(-1).cpu().numpy().astype(np.float32)
        return np.clip(rul, 0.0, float(self.max_rul))

    # ── Internal helpers ───────────────────────────────────────────────

    def _normalise(self, raw_window: np.ndarray) -> np.ndarray:
        """Apply training-time Min-Max normalisation to a raw sensor window.

        ``raw_window`` is in physical units; output is clipped to [0,1],
        matching ``data_loader._normalise_sensors``.  Requires the scaler
        JSON to have been loaded — raises if missing.
        """
        if self._scaler is None:
            raise RuntimeError(
                f"Cannot normalise raw input: scaler {self._scaler_path} not "
                f"loaded. Run experiments/rul/export_scaler.py to generate it, "
                f"or call predict_rul(..., raw=False) with pre-normalised input."
            )
        mins = np.asarray(self._scaler["min"], dtype=np.float32)
        ranges = np.asarray(self._scaler["range"], dtype=np.float32)
        norm = (raw_window - mins) / ranges
        return np.clip(norm, 0.0, 1.0).astype(np.float32)

    def _prepare_window(
        self, window: np.ndarray, raw: bool = True
    ) -> np.ndarray:
        """Ensure window is (window_size, n_sensors) float32, padded/truncated.

        When ``raw=True`` the input is treated as physical-unit readings and
        Min-Max normalised to [0,1] before returning (matching training).
        When ``raw=False`` the input is assumed already normalised.
        """
        w = np.asarray(window, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(
                f"Expected 2-D window (window_size, n_sensors), got shape {w.shape}"
            )
        ws = self._config.get("window_size", DEFAULT_WINDOW_SIZE)
        ns = self._config.get("n_sensors", N_SENSORS)
        if w.shape[1] != ns:
            raise ValueError(
                f"Expected {ns} sensors, got {w.shape[1]}. "
                f"Sensor count mismatch — check preprocessing."
            )
        if w.shape[0] < ws:
            # Pad at the front with the earliest row (short engine).
            pad = np.tile(w[:1], (ws - w.shape[0], 1))
            w = np.vstack([pad, w])
        elif w.shape[0] > ws:
            w = w[-ws:]  # take the most recent window_size cycles
        # Normalise last so padding uses raw values (matches data_loader,
        # which normalises the whole engine array then builds windows).
        if raw:
            w = self._normalise(w)
        return w.astype(np.float32)
