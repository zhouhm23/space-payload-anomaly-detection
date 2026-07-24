"""L1 rule — physical-expectation (setpoint) detection.

Complements the statistical L1 rules (sigma / iqr / rate) with a
physically-grounded check: the scientist tells the system what the
signal *should* look like, and this rule flags deviations from that
expectation.  Three modes cover the common sensor archetypes:

  command    — command/status channel: one normal value, one or more
               anomaly values (e.g. switch ``0 = normal``, ``1 = fault``).
  range      — constant monitoring: expected working point +/- tolerance
               (e.g. thermostat ``25 C +/- 2 C``); also accepts an
               explicit ``[range_low, range_high]`` pair.
  enumerate  — discrete states: a fixed set of legal values
               (e.g. gear position ``0/1/2/3``).

Unlike the statistical L1 rules this one is **opt-in**: it has no
sensible default behaviour without scientist-provided expected values,
so it is deliberately excluded from :data:`DEFAULT_L1_MODULES`.
Per-channel configuration (Stage-2) and the @command DSL (Plan 3) opt
in explicitly via ``@算法=l1_setpoint`` plus the required parameters.

Decision semantics: out-of-spec samples yield ``DECISION_ALERT`` (not
``SKIP``).  ``SKIP`` is reserved for the constant-channel / data-quality
short-circuit owned by :mod:`l1_constant`; setpoint is "the value is
wrong" rather than "the channel is unusable", so the cascade still runs
L2/L3 to corroborate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base_filter import BaseFilter
from ..cascade_types import (
    LayerResult,
    LAYER_L1_CLASSIC,
    DECISION_PASS,
    DECISION_ALERT,
)
from ._base import register_filter

__all__ = ["L1SetpointRule"]


# Score assigned to an outright anomaly in command / enumerate mode.
# Matches the L1 "alert" severity band used by sigma (0.8) / iqr (0.7);
# setpoint is a hard physical violation so it gets the top of that band.
SETPOINT_ANOMALY_SCORE = 1.0

# Default score for a command-mode "middle" value (not command, not anomaly).
# Command/status channels rarely have legitimate intermediate states, so an
# unexpected value is flagged as a mild warning rather than a hard alert.
DEFAULT_MID_STATE_SCORE = 0.5

# Tolerance below which two floats are considered equal in command /
# enumerate mode.  Kept generous (1e-6) so integer-valued sensor channels
# compared against floats still match robustly; tight enough that genuine
# analogue drift is caught.
VALUE_EQUAL_ATOL = 1e-6


@register_filter("l1_setpoint")
class L1SetpointRule(BaseFilter):
    """Detect deviations from a scientist-supplied physical expectation.

    Args:
        mode: one of ``{"command", "range", "enumerate"}``.

        command_value: (command mode) the normal / quiescent value.
        anomaly_values: (command mode) values that indicate a fault.

        expected: (range mode) the nominal working point.
        tolerance: (range mode) half-width of the acceptable band.
        range_low / range_high: (range mode) alternative to
            ``expected`` / ``tolerance`` — specify the band directly.

        legal_values: (enumerate mode) the set of allowed discrete values.

        mid_state_score: (command mode) score assigned to a value that is
            neither the command value nor an explicit anomaly value.
            Defaults to ``0.5`` (mild warning).

    Construction validates that the parameters required by ``mode`` are
    present and raises ``ValueError`` otherwise.
    """

    name = "l1_setpoint"
    layer = LAYER_L1_CLASSIC

    VALID_MODES = {"command", "range", "enumerate"}

    def __init__(
        self,
        *,
        mode: str = "range",
        # command mode
        command_value: float | None = None,
        anomaly_values: list[float] | None = None,
        # range mode
        expected: float | None = None,
        tolerance: float | None = None,
        range_low: float | None = None,
        range_high: float | None = None,
        # enumerate mode
        legal_values: list[float] | None = None,
        # scoring
        mid_state_score: float = DEFAULT_MID_STATE_SCORE,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(self.VALID_MODES)!r}, got {mode!r}"
            )
        self.mode = mode

        # Normalise the optional list/tuple params to lists up front so the
        # per-mode validation and filter() paths don't have to keep handling
        # None / tuple / list variants.
        self.anomaly_values = (
            [float(x) for x in anomaly_values] if anomaly_values is not None else None
        )
        self.legal_values = (
            [float(x) for x in legal_values] if legal_values is not None else None
        )

        self.command_value = (
            float(command_value) if command_value is not None else None
        )
        self.expected = float(expected) if expected is not None else None
        self.tolerance = float(tolerance) if tolerance is not None else None
        self.range_low = float(range_low) if range_low is not None else None
        self.range_high = float(range_high) if range_high is not None else None

        # mid_state_score only matters for command mode but is cheap to validate
        # unconditionally — guards against a nonsensical custom value later.
        if not (0.0 <= float(mid_state_score) <= 1.0):
            raise ValueError(
                f"mid_state_score must be in [0, 1], got {mid_state_score!r}"
            )
        self.mid_state_score = float(mid_state_score)

        self._validate_args()

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def _validate_args(self) -> None:
        """Check the parameters required by ``self.mode`` are present.

        Raises ``ValueError`` with a mode-specific message so the @command
        DSL validator can surface a useful error when a scientist writes
        ``@算法=l1_setpoint`` without the matching ``@参数.*`` clauses.
        """
        if self.mode == "command":
            if self.command_value is None:
                raise ValueError(
                    "command mode requires 'command_value' "
                    "(the normal/quiescent value)"
                )
            if self.anomaly_values is None or len(self.anomaly_values) == 0:
                raise ValueError(
                    "command mode requires a non-empty 'anomaly_values' list"
                )

        elif self.mode == "range":
            has_expected = self.expected is not None and self.tolerance is not None
            has_pair = self.range_low is not None and self.range_high is not None
            if not (has_expected or has_pair):
                raise ValueError(
                    "range mode requires either ('expected', 'tolerance') "
                    "or ('range_low', 'range_high')"
                )
            if has_expected and self.tolerance is not None and self.tolerance < 0:
                raise ValueError(
                    f"tolerance must be non-negative, got {self.tolerance!r}"
                )
            if has_pair and self.range_low is not None and self.range_high is not None:
                if self.range_low > self.range_high:
                    raise ValueError(
                        f"range_low ({self.range_low}) must be <= "
                        f"range_high ({self.range_high})"
                    )

        else:  # enumerate
            if self.legal_values is None or len(self.legal_values) == 0:
                raise ValueError(
                    "enumerate mode requires a non-empty 'legal_values' list"
                )

    # ------------------------------------------------------------------
    # Range-mode geometry helpers
    # ------------------------------------------------------------------

    def _range_bounds(self) -> tuple[float, float]:
        """Return the inclusive ``[lo, hi]`` acceptable band for range mode.

        Combines the ``expected``/``tolerance`` and ``range_low``/``range_high``
        specifications into a single pair.  When both are given the explicit
        ``range_low``/``range_high`` pair wins (matches DSL intent: an
        explicit band is more specific than a centre +/- half-width).
        """
        if self.range_low is not None and self.range_high is not None:
            return float(self.range_low), float(self.range_high)
        # _validate_args guarantees expected & tolerance are both set here.
        assert self.expected is not None and self.tolerance is not None
        return (
            float(self.expected) - float(self.tolerance),
            float(self.expected) + float(self.tolerance),
        )

    def _range_center(self) -> float:
        lo, hi = self._range_bounds()
        return (lo + hi) / 2.0

    def _range_half_width(self) -> float:
        lo, hi = self._range_bounds()
        return max((hi - lo) / 2.0, 1e-9)

    # ------------------------------------------------------------------
    # Serialization helper (used in detail for the alert message)
    # ------------------------------------------------------------------

    def _spec_dict(self) -> dict[str, Any]:
        """Compact JSON-friendly description of the active setpoint spec."""
        if self.mode == "command":
            return {
                "mode": "command",
                "command_value": self.command_value,
                "anomaly_values": list(self.anomaly_values or []),
            }
        if self.mode == "range":
            lo, hi = self._range_bounds()
            return {
                "mode": "range",
                "range_low": lo,
                "range_high": hi,
            }
        return {
            "mode": "enumerate",
            "legal_values": list(self.legal_values or []),
        }

    # ------------------------------------------------------------------
    # BaseFilter contract
    # ------------------------------------------------------------------

    def filter(
        self,
        values: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> LayerResult:
        v = np.asarray(values, dtype=np.float64).ravel()
        n = len(v)

        # Empty input — return all-zero per-sample score and PASS.  Unlike
        # l1_constant we do NOT short-circuit with SKIP: setpoint is a value
        # check, an empty block has no values to check, so it cleanly passes.
        if n == 0:
            return LayerResult(
                layer=LAYER_L1_CLASSIC,
                decision=DECISION_PASS,
                score=0.0,
                detail={
                    "rules": [],
                    "per_sample_score": np.zeros(0, dtype=np.float32),
                    "setpoint_mode": self.mode,
                    "setpoint_spec": self._spec_dict(),
                },
            )

        per_sample = np.zeros(n, dtype=np.float32)
        finite_mask = np.isfinite(v)
        rules: list[str] = []
        decision = DECISION_PASS

        if self.mode == "command":
            decision = self._eval_command(v, finite_mask, per_sample, rules)
        elif self.mode == "range":
            decision = self._eval_range(v, finite_mask, per_sample, rules)
        else:  # enumerate
            decision = self._eval_enumerate(v, finite_mask, per_sample, rules)

        rep_score = float(per_sample.max()) if n > 0 else 0.0
        return LayerResult(
            layer=LAYER_L1_CLASSIC,
            decision=decision,
            score=rep_score,
            detail={
                "rules": rules,
                "per_sample_score": per_sample,
                "setpoint_mode": self.mode,
                "setpoint_spec": self._spec_dict(),
            },
        )

    # ------------------------------------------------------------------
    # Per-mode evaluators (mutate per_sample / rules in place for speed)
    # ------------------------------------------------------------------

    def _eval_command(
        self,
        v: np.ndarray,
        finite_mask: np.ndarray,
        per_sample: np.ndarray,
        rules: list[str],
    ) -> str:
        """command mode: command_value -> 0, anomaly_values -> 1.0, else mid."""
        assert self.command_value is not None
        assert self.anomaly_values is not None

        is_cmd = finite_mask & np.isclose(
            v, self.command_value, atol=VALUE_EQUAL_ATOL, rtol=0.0, equal_nan=False
        )
        is_anom = np.zeros(len(v), dtype=bool)
        for av in self.anomaly_values:
            is_anom |= finite_mask & np.isclose(
                v, av, atol=VALUE_EQUAL_ATOL, rtol=0.0, equal_nan=False
            )

        # Anomaly has priority over command (a value can't be both, but if
        # atol ever caused overlap we want the more severe reading).
        per_sample[is_anom] = SETPOINT_ANOMALY_SCORE
        mid = finite_mask & ~is_cmd & ~is_anom
        per_sample[mid] = self.mid_state_score

        if bool(is_anom.any()):
            rules.append("setpoint_command_anomaly")
            return DECISION_ALERT
        if bool(mid.any()) and self.mid_state_score > 0.0:
            # Unexpected middle state is a warning, not a hard alert — but
            # still surface the rule so consumers know setpoint flagged it.
            rules.append("setpoint_command_mid_state")
            return DECISION_ALERT
        return DECISION_PASS

    def _eval_range(
        self,
        v: np.ndarray,
        finite_mask: np.ndarray,
        per_sample: np.ndarray,
        rules: list[str],
    ) -> str:
        """range mode: in-band -> 0, out-of-band -> linear ramp up to 1.0.

        Ramp shape: a sample exactly on the boundary scores 0; one full
        half-width beyond the boundary scores 1.0; between the two the
        score grows linearly.  This makes the score proportional to how
        far outside the band the sample lands, which is what the alert
        message wants to convey.
        """
        lo, hi = self._range_bounds()
        center = self._range_center()
        half_w = self._range_half_width()

        in_band = finite_mask & (v >= lo) & (v <= hi)
        # per_sample already zero — in_band samples stay at 0.

        out = finite_mask & ~in_band
        if bool(out.any()):
            # Per-sample score grows linearly with how far past the band
            # edge the sample lands: 0 at the boundary, 1.0 one half-width
            # beyond it, capped at 1.0.  Distance is measured from the band
            # centre so the ramp is symmetric on both sides.
            dist_from_center = np.abs(v[out] - center)
            excess = np.maximum(dist_from_center - half_w, 0.0)
            per_sample[out] = np.minimum(1.0, excess / max(half_w, 1e-9)).astype(
                np.float32
            )
            rules.append("setpoint_range_violation")
            return DECISION_ALERT
        return DECISION_PASS

    def _eval_enumerate(
        self,
        v: np.ndarray,
        finite_mask: np.ndarray,
        per_sample: np.ndarray,
        rules: list[str],
    ) -> str:
        """enumerate mode: legal_values -> 0, anything else finite -> 1.0."""
        assert self.legal_values is not None

        is_legal = np.zeros(len(v), dtype=bool)
        for lv in self.legal_values:
            is_legal |= finite_mask & np.isclose(
                v, lv, atol=VALUE_EQUAL_ATOL, rtol=0.0, equal_nan=False
            )

        illegal = finite_mask & ~is_legal
        per_sample[illegal] = SETPOINT_ANOMALY_SCORE
        if bool(illegal.any()):
            rules.append("setpoint_enumerate_illegal")
            return DECISION_ALERT
        return DECISION_PASS
