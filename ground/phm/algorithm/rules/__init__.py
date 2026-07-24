"""Modular cascade rule library — L1 classic checks and L3 physical checks.

Each rule is an independent :class:`BaseFilter` subclass registered under a
stable string name in :data:`FILTER_REGISTRY`.  This lets per-channel
configurations compose their own L1 / L3 chains by name (Stage-2 work)
without the cascade hard-coding which checks run.

The two combinator classes :class:`ClassicFilter` and
:class:`PhysicalConstraint` consume these modules internally while keeping
their pre-refactor constructor signatures and ``filter()`` return shapes
byte-for-byte identical (Stage-1 = pure structural refactor, zero behaviour
change — verified by ``tests/test_rules_equivalence.py``).

Module layout::

    _base.py             FILTER_REGISTRY + register_filter + build_filter
    l1_constant.py       L1ConstantRule   (was ClassicFilter rule 1)
    l1_sigma.py          L1SigmaRule      (was ClassicFilter rule 2)
    l1_iqr.py            L1IqrRule        (was ClassicFilter rule 3)
    l1_rate.py           L1RateRule       (was ClassicFilter rule 4)
    l1_setpoint.py       L1SetpointRule   (opt-in expert module, Phase 1.5)
    l3_nan_sanitise.py   L3NanSanitiseRule (was PhysicalConstraint rule 1)
    l3_constant.py       L3ConstantRule    (was PhysicalConstraint rule 2)
    l3_range.py          L3RangeRule       (was PhysicalConstraint rule 3)
    l3_rate.py           L3RateRule        (was PhysicalConstraint rule 4)
    l3_variance.py       L3VarianceRule    (was PhysicalConstraint rule 5)

``l1_setpoint`` is registered but **deliberately not in
DEFAULT_L1_MODULES**: it is an opt-in expert module that requires a
scientist-supplied physical expectation (command value / expected range /
legal values) and has no sensible default behaviour.
"""

from __future__ import annotations

# Importing the rule modules has the side effect of populating
# FILTER_REGISTRY via the @register_filter decorator.  Keep these imports
# in the same order as the rule numbers above so the registry's insertion
# order matches the canonical cascade order (matters for default chains
# and for diagnostic listings).
from . import _base  # noqa: F401  (ensures registry module loaded first)
from ._base import FILTER_REGISTRY, register_filter, build_filter, FilterConfig
from .l1_constant import L1ConstantRule
from .l1_sigma import L1SigmaRule
from .l1_iqr import L1IqrRule
from .l1_rate import L1RateRule
# l1_setpoint is an opt-in expert module (Phase 1.5).  It is registered in
# FILTER_REGISTRY but intentionally excluded from DEFAULT_L1_MODULES — it
# requires scientist-supplied expected values and has no sensible default.
from .l1_setpoint import L1SetpointRule
from .l3_nan_sanitise import L3NanSanitiseRule
from .l3_constant import L3ConstantRule
from .l3_range import L3RangeRule
from .l3_rate import L3RateRule
from .l3_variance import L3VarianceRule
# Layer 3.5 leak-free post-processing modules (Day26-续).  These wrap the
# validated knee / EMA / persistence primitives as BaseFilter modules so
# they can be registered and composed like the L1/L3 rules above.  They
# are stateful (per-channel) — use their ``filter_channel`` method for
# streaming; the base ``filter`` falls back to a single-channel key.
from .l3_knee_threshold import L3KneeThresholdRule
from .l3_ema_smoothing import L3EmaSmoothingRule
from .l3_persistence import L3PersistenceRule

__all__ = [
    # Registry
    "FILTER_REGISTRY",
    "register_filter",
    "build_filter",
    "FilterConfig",
    # L1 rule modules (classic statistical checks)
    "L1ConstantRule",
    "L1SigmaRule",
    "L1IqrRule",
    "L1RateRule",
    # Opt-in L1 expert module (Phase 1.5; not in DEFAULT_L1_MODULES)
    "L1SetpointRule",
    # L3 rule modules (physical-constraint post-processing)
    "L3NanSanitiseRule",
    "L3ConstantRule",
    "L3RangeRule",
    "L3RateRule",
    "L3VarianceRule",
    # Layer 3.5 leak-free post-processing modules
    "L3KneeThresholdRule",
    "L3EmaSmoothingRule",
    "L3PersistenceRule",
    # Default module-name chains — the names correspond to the
    # pre-refactor ClassicFilter / PhysicalConstraint behaviour (all rules
    # enabled, default thresholds).  Stage-2 configs can reference these
    # lists instead of hard-coding names.
    "DEFAULT_L1_MODULES",
    "DEFAULT_L3_MODULES",
    "DEFAULT_L35_MODULES",
]


# Canonical default chains (by registry name).  Order matches the original
# rule evaluation order in ClassicFilter / PhysicalConstraint — the
# combinators rely on this order for severity aggregation (σ > IQR > rate
# for L1) and for the L3 early-return semantics (constant suppression
# short-circuits rules 3/4/5).
DEFAULT_L1_MODULES: list[str] = [
    "l1_constant",
    "l1_sigma",
    "l1_iqr",
    "l1_rate",
]
DEFAULT_L3_MODULES: list[str] = [
    "l3_nan_sanitise",
    "l3_constant",
    "l3_range",
    "l3_rate",
    "l3_variance",
]
# Layer 3.5 default chain — the order is the validated signal flow from
# ``experiments/metrics/run_ablation_a6.py``: knee threshold (derive
# operating point) → EMA smoothing (remove block seams) → persistence
# (temporal de-jitter).  All three are stateful; composing them in a
# PhysicalConstraint ``rules=`` chain is possible but the stateful
# ``filter_channel`` path is what online integration will use.
DEFAULT_L35_MODULES: list[str] = [
    "l3_knee_threshold",
    "l3_ema_smoothing",
    "l3_persistence",
]
