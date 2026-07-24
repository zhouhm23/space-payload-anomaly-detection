"""Registry + base helpers for modular cascade filter rules.

This is the registration mechanism that lets each L1 / L3 rule live in its
own module while still being discoverable by name (so per-channel configs
can say ``"l1_modules": ["l1_sigma", "l1_rate"]`` and the cascade can build
the chain without importing every concrete class).

Mirrors the pattern of ``_registry.MODEL_REGISTRY`` but for filter plugins:
a decorator writes into a module-level dict and a ``build_filter`` factory
instantiates by name.
"""

from __future__ import annotations

from typing import Any, Callable

from ..base_filter import BaseFilter

__all__ = [
    "FILTER_REGISTRY",
    "register_filter",
    "build_filter",
    "FilterConfig",
]


# Module-level registry: name → filter class.  Populated by the
# ``@register_filter("name")`` decorator on each concrete rule module.
FILTER_REGISTRY: dict[str, type[BaseFilter]] = {}


def register_filter(name: str) -> Callable[[type[BaseFilter]], type[BaseFilter]]:
    """Class decorator: register ``cls`` under ``name``.

    Usage::

        @register_filter("l1_sigma")
        class L1SigmaRule(BaseFilter): ...
    """

    def _deco(cls: type[BaseFilter]) -> type[BaseFilter]:
        if not isinstance(name, str) or not name:
            raise ValueError("filter name must be a non-empty string")
        FILTER_REGISTRY[name] = cls
        # Keep the registered name on the class as a convenience for
        # debugging / serialisation, without overriding an explicit class
        # attribute (tests rely on the hard-coded ``name`` attr).
        cls._registry_name = name
        return cls

    return _deco


def build_filter(name: str, **kwargs: Any) -> BaseFilter:
    """Instantiate a registered filter by name.

    Raises ``KeyError`` if ``name`` is not in :data:`FILTER_REGISTRY`.
    """
    cls = FILTER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"unknown filter module: {name!r}")
    return cls(**kwargs)


class FilterConfig:
    """Marker base class for rule-module config objects.

    Rule modules that carry tunable thresholds subclass this so configs can
    be passed around uniformly.  It deliberately adds no behaviour — the
    concrete rule classes keep accepting plain kwargs too, so existing
    callers (e.g. ``ClassicFilter(sigma_k=3.0)``) stay unchanged.
    """

    pass
