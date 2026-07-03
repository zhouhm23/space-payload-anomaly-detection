"""Space-segment preprocessing pipeline.

Receives raw sensor data (possibly with NaN) and produces model-ready input
for TSPulse anomaly detection.

Pipeline stages (format conversion only — no denoising/filtering, so anomaly
features are preserved):
  1. Missing-value imputation  — linear interpolation, edge-fill
  2. Normalization             — StandardScaler (fit on training data only)

No filtering is applied.  Filtering would remove the very anomalies that
the detector is trying to find.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler


@dataclass
class SpacePreprocessor:
    """On-orbit preprocessing: impute NaN + standardize.

    Only converts data format (imputation + normalization). Does NOT alter
    signal characteristics — no low-pass, no median, no denoising.
    """

    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)

    # -- public API --

    def fit(self, train_values: np.ndarray) -> "SpacePreprocessor":
        """Fit the normalizer on training data (after imputation)."""
        clean = self._impute(train_values.astype(np.float64))
        self._scaler = StandardScaler().fit(clean.reshape(-1, 1))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Impute NaN + standardize. Returns 1-D float32."""
        if self._scaler is None:
            raise RuntimeError("Call .fit() first, or use fit_transform().")
        x = self._impute(values.astype(np.float64))
        x = self._scaler.transform(x.reshape(-1, 1)).flatten()
        return x.astype(np.float32)

    def fit_transform(
        self, values: np.ndarray, train_values: np.ndarray | None = None
    ) -> np.ndarray:
        if train_values is not None:
            self.fit(train_values)
        else:
            clean = self._impute(values.astype(np.float64))
            self._scaler = StandardScaler().fit(clean.reshape(-1, 1))
        return self.transform(values)

    # -- internal --

    @staticmethod
    def _impute(x: np.ndarray) -> np.ndarray:
        """Linear interpolation for NaN, edge-fill at boundaries."""
        n = len(x)
        mask = np.isnan(x)
        if not mask.any():
            return x
        valid_idx = np.where(~mask)[0]
        if len(valid_idx) == 0:
            return np.zeros(n)
        x[mask] = np.interp(
            np.where(mask)[0].astype(float),
            valid_idx.astype(float),
            x[valid_idx],
            left=x[valid_idx[0]],
            right=x[valid_idx[-1]],
        )
        return x
