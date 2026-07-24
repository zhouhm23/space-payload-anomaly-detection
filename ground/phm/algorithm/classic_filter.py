"""Layer 1 — Classic statistical filter (fast pre-screening).

Goal: low-cost screening that catches obviously-anomalous or obviously-broken
data *before* the expensive DL detector runs.  Four independent checks:

1. **Constant-channel detection** — ``std < ε`` ⇒ the channel carries no
   information (e.g. TSB-UAD channels C-2/D-14/M-6/S-2/T-5 whose analog
   dimension is pinned by a command bit).  Returning ``skip`` tells the
   cascade to force score=0 and avoid wasting a TSPulse forward pass.

2. **3σ rule** — any sample beyond ``mean ± k·std`` is flagged ``alert``.

3. **IQR rule** — any sample outside ``[Q1−1.5·IQR, Q3+1.5·IQR]``.

4. **Rate-of-change limit** — consecutive-sample jump exceeds a threshold.

Checks are individually toggleable so the filter can be tailored per
deployment.
"""

from __future__ import annotations

import numpy as np

from .base_filter import BaseFilter
from .cascade_types import (
    LayerResult,
    LAYER_L1_CLASSIC,
    DECISION_PASS,
    DECISION_ALERT,
    DECISION_SKIP,
    DECISION_SUSPICIOUS,
)

__all__ = ["ClassicFilter"]


class ClassicFilter(BaseFilter):
    """Layer-1 statistical pre-filter.

    Args:
        enable_constant:   detect near-constant channels.
        enable_sigma:      3σ outlier check.
        enable_iqr:        IQR outlier check.
        enable_rate:       rate-of-change check.
        constant_std:      channels with std below this are treated as constant.
        sigma_k:           number of standard deviations for the σ rule.
        iqr_factor:        multiplier for the IQR fence (classic 1.5).
        rate_quantile:     if ``max_rate`` is None it is derived from the
                           data as this quantile of absolute diffs (default
                           p99 → only extreme jumps trigger).
        rate_multiplier:   ``max_rate = p99 * multiplier``.
        max_rate:          explicit rate-of-change threshold.  Overrides the
                           quantile derivation when not None.
    """

    name = "classic_filter"

    def __init__(
        self,
        *,
        enable_constant: bool = True,
        enable_sigma: bool = True,
        enable_iqr: bool = True,
        enable_rate: bool = True,
        constant_std: float = 1e-3,
        sigma_k: float = 3.0,
        iqr_factor: float = 1.5,
        rate_quantile: float = 99.0,
        rate_multiplier: float = 5.0,
        max_rate: float | None = None,
    ) -> None:
        self.enable_constant = enable_constant
        self.enable_sigma = enable_sigma
        self.enable_iqr = enable_iqr
        self.enable_rate = enable_rate
        self.constant_std = constant_std
        self.sigma_k = sigma_k
        self.iqr_factor = iqr_factor
        self.rate_quantile = rate_quantile
        self.rate_multiplier = rate_multiplier
        self.max_rate = max_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        """Evaluate all enabled checks on *values*.

        Returns a single :class:`LayerResult`.  The ``decision`` is the most
        severe finding (skip > alert > suspicious > pass).  ``detail`` lists
        every rule that triggered.
        """
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)
        rules: list[str] = []
        decision = DECISION_PASS
        per_sample = np.zeros(n, dtype=np.float32)

        if n == 0:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=DECISION_SKIP,
                score=0.0,
                detail={"reason": "empty_input"},
            )

        # NaN / Inf safety
        finite_mask = np.isfinite(v)
        n_finite = int(finite_mask.sum())

        # --- 1. Constant-channel detection ------------------------------
        if self.enable_constant and n_finite >= 2:
            finite_vals = v[finite_mask]
            std = float(np.std(finite_vals))
            if std < self.constant_std:
                return LayerResult(
                    layer=LAYER_L1_CLASSIC,
                    decision=DECISION_SKIP,
                    score=0.0,
                    detail={
                        "rules": ["constant_channel"],
                        "std": std,
                        "threshold": self.constant_std,
                        "n_samples": n,
                    },
                )

        # If almost everything is NaN/Inf, skip — data is unusable.
        if n_finite < 2:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=DECISION_SKIP,
                score=0.0,
                detail={
                    "rules": ["insufficient_finite"],
                    "n_finite": n_finite,
                    "n_samples": n,
                },
            )

        clean = v[finite_mask]
        mu = float(np.mean(clean))
        sigma = float(np.std(clean))

        # --- 2. 3σ rule -------------------------------------------------
        if self.enable_sigma and sigma > 0:
            lo = mu - self.sigma_k * sigma
            hi = mu + self.sigma_k * sigma
            # NOTE: parentheses are required here because ``&`` binds
            # tighter than ``|`` in Python.  Writing
            # ``finite_mask & (v < lo) | (v > hi)`` would parse as
            # ``(finite_mask & (v < lo)) | (v > hi)`` and incorrectly flag
            # +Inf samples (v > hi is True for +Inf even though finite_mask
            # is False).  See experiments/diag/reproduce_classic_filter_opbug.py.
            sigma_out = finite_mask & ((v < lo) | (v > hi))
            n_sigma = int(sigma_out.sum())
            if n_sigma > 0:
                rules.append("sigma_3")
                per_sample[sigma_out] = np.maximum(per_sample[sigma_out], 0.8)
                if decision == DECISION_PASS:
                    decision = DECISION_ALERT

        # --- 3. IQR rule ------------------------------------------------
        if self.enable_iqr and n_finite >= 4:
            q1 = float(np.percentile(clean, 25))
            q3 = float(np.percentile(clean, 75))
            iqr = q3 - q1
            if iqr > 0:
                lo_iqr = q1 - self.iqr_factor * iqr
                hi_iqr = q3 + self.iqr_factor * iqr
                # See note on the sigma rule above: parentheses required so
                # non-finite samples are excluded by finite_mask before the
                # OR combines the two comparisons.
                iqr_out = finite_mask & ((v < lo_iqr) | (v > hi_iqr))
                n_iqr = int(iqr_out.sum())
                if n_iqr > 0:
                    rules.append("iqr")
                    per_sample[iqr_out] = np.maximum(per_sample[iqr_out], 0.7)
                    if decision == DECISION_PASS:
                        decision = DECISION_ALERT

        # --- 4. Rate-of-change ------------------------------------------
        if self.enable_rate and n >= 2:
            diffs = np.abs(np.diff(v))
            finite_diffs = diffs[np.isfinite(diffs)]
            if len(finite_diffs) > 0:
                if self.max_rate is not None:
                    thr = self.max_rate
                else:
                    p = float(np.percentile(finite_diffs, self.rate_quantile))
                    thr = p * self.rate_multiplier
                if thr > 0:
                    rate_out = np.zeros(n, dtype=bool)
                    rate_out[1:] = diffs > thr
                    n_rate = int(rate_out.sum())
                    if n_rate > 0:
                        rules.append("rate_of_change")
                        per_sample[rate_out] = np.maximum(per_sample[rate_out], 0.6)
                        if decision == DECISION_PASS:
                            decision = DECISION_SUSPICIOUS

        rep_score = float(np.max(per_sample)) if n > 0 else 0.0
        return LayerResult(
            layer=LAYER_L1_CLASSIC,
            decision=decision,
            score=rep_score,
            detail={
                "rules": rules,
                "mean": mu,
                "std": sigma,
                "n_samples": n,
                "per_sample_score": per_sample,
            },
        )


__all__ = ["ClassicFilter"]
