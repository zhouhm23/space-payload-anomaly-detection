"""Layer-1 classic statistical filter (space-segment lightweight version).

Standalone module — does NOT import from the ground ``phm`` package, so it
can run on the resource-constrained space segment without pulling in the
full PHM stack.

Only implements the checks needed for space-side pre-filtering:

1. **Constant-channel detection** — ``std < ε`` ⇒ skip TSPulse entirely.
   This is the primary optimisation: TSB-UAD constant channels (C-2/D-14/
   M-6/S-2/T-5) waste a full forward pass producing meaningless scores.

2. **3σ / IQR outlier flag** — included for completeness but the space
   segment currently only uses the constant-channel skip.

Returns a plain ``(decision: str, detail: dict)`` tuple — no dependency on
cascade_types / LayerResult.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SpaceClassicFilter"]

# Decision constants (kept in sync with ground phm.algorithm.cascade_types)
DECISION_PASS = "pass"
DECISION_SKIP = "skip"
DECISION_ALERT = "alert"


class SpaceClassicFilter:
    """Lightweight L1 filter for the space segment.

    Args:
        constant_std: channels with std below this are flagged ``skip``.
        sigma_k:      σ multiplier for the 3σ rule.
    """

    def __init__(
        self,
        *,
        constant_std: float = 1e-3,
        sigma_k: float = 3.0,
    ) -> None:
        self.constant_std = constant_std
        self.sigma_k = sigma_k

    def check(self, values: np.ndarray) -> tuple[str, dict]:
        """Return ``(decision, detail)`` for a 1-D telemetry block.

        ``decision`` is one of ``"skip"`` / ``"alert"`` / ``"pass"``.
        When ``"skip"`` is returned the caller should force scores=None and
        avoid running the DL detector.
        """
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)
        if n == 0:
            return DECISION_SKIP, {"reason": "empty_input", "n": 0}

        finite_mask = np.isfinite(v)
        n_finite = int(finite_mask.sum())
        if n_finite < 2:
            return DECISION_SKIP, {
                "reason": "insufficient_finite",
                "n_finite": n_finite,
                "n": n,
            }

        clean = v[finite_mask]
        std = float(np.std(clean))

        # 1. Constant channel — primary space-side optimisation
        if std < self.constant_std:
            return DECISION_SKIP, {
                "reason": "constant_channel",
                "std": std,
                "threshold": self.constant_std,
                "n": n,
            }

        # 2. 3σ quick check (informational — space segment does not act on it)
        mu = float(np.mean(clean))
        sigma_out = 0
        if std > 0:
            lo = mu - self.sigma_k * std
            hi = mu + self.sigma_k * std
            sigma_out = int(np.sum((v < lo) | (v > hi)))

        detail = {"std": std, "mean": mu, "n_sigma_outliers": sigma_out, "n": n}
        if sigma_out > 0:
            return DECISION_ALERT, detail
        return DECISION_PASS, detail
