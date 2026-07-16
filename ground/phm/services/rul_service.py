"""RUL (Remaining Useful Life) degradation-prediction service.

Wraps the product-library :class:`RULPredictor` (LSTM+Attention trained on
NASA C-MAPSS) into a service that the ground system polls for long-horizon
degradation forecasts.  During development the data source is the C-MAPSS
benchmark itself (real turbofan degradation trajectories, cycle-scale) —
this mirrors how TSPulse is evaluated on TSB-UAD and TTM-R3 on NASA MSL:
each foundation model is demonstrated on its own validated benchmark.

**Channel opt-in via ``@rul:<model>`` tags.**  A sensor node's ``description``
field may carry a tag such as ``@rul:fd001`` to opt the channel into RUL
prediction with the named model.  This is a soft convention — no schema
change, full backward compatibility (old ``device_config.json`` files keep
working; channels without the tag are simply skipped).

**Data-source abstraction.**  :class:`RulDataSource` is the protocol;
:class:`CMAPSSDataSource` is the development-time implementation.  When real
payload degradation sensors come online, implement the protocol against the
live telemetry store and swap it in via :func:`deps.init` — no service code
change needed.

Architecture (development-time)::

    datasets/CMAPSSData/   (bypass — not the space-segment telemetry stream)
        ↓
    CMAPSSDataSource
        ↓
    RulService.predict_all()  →  /api/rul  →  monitor.js RUL panel

The MSL channels (C-1/D-14) keep flowing through the space→ground telemetry
path for TSPulse detection and TTM-R3 forecasting, untouched.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..algorithm import RULPredictor
from .config_service import ConfigService

logger = logging.getLogger(__name__)

__all__ = ["RulDataSource", "CMAPSSDataSource", "RulService"]


# ── Channel-tag parsing ────────────────────────────────────────────────────
# Matches "@rul:fd001" / "@rul:FD003" etc. inside a sensor description.
# Case-insensitive; the captured group is lower-cased to match the model id
# used as the RULPredictor subset and the predictors dict key.
_RUL_TAG = re.compile(r"@rul:\s*(fd00[1234])", re.IGNORECASE)


def parse_rul_tag(description: str | None) -> str | None:
    """Return the lower-cased model id from a ``@rul:fd00X`` tag, or None.

    Only the first match is used; if a description accidentally carries two
    tags the first wins (deterministic).
    """
    if not description:
        return None
    m = _RUL_TAG.search(description)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------
# Data source protocol + C-MAPSS implementation
# ---------------------------------------------------------------------------

class RulDataSource(Protocol):
    """Pluggable data source for RUL prediction.

    The development implementation (:class:`CMAPSSDataSource`) replays the
    C-MAPSS test trajectories.  A future real-payload implementation would
    pull windows from the RingBuffer / SQLiteStore for channels that carry
    multivariate degradation sensors.
    """

    def channels(self) -> list[str]:
        """Channel ids this source can serve windows for."""
        ...

    def get_window(self, channel: str, n_cycles: int) -> np.ndarray | None:
        """Return a ``(n_cycles, n_sensors)`` raw-value window, or None.

        ``None`` signals "not enough data yet" — the caller skips the channel
        for this poll.  The returned array is in **physical units**; the
        RULPredictor normalises internally.
        """
        ...

    def advance(self) -> None:
        """Advance the internal playback pointer by one cycle."""
        ...


# C-MAPSS 26-column layout (kept here so this module is self-contained and
# does not import from the private experiments/ package).
_CMAPSS_COLS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


class CMAPSSDataSource:
    """Replay C-MAPSS test trajectories cycle-by-cycle.

    Loads ``test_FD00X.txt`` once, keeps a per-engine cycle pointer, and on
    each :meth:`advance` bumps every engine forward by one cycle.  Windows
    are the last ``n_cycles`` rows up to the current pointer (front-padded
    with the first row if the engine hasn't reached ``n_cycles`` yet).  When
    an engine runs out of data its pointer resets to the start so the demo
    loops indefinitely.

    Channel ids follow ``CMAPSS_<SUBSET>_<unit>`` (e.g. ``CMAPSS_FD001_1``),
    matching the ``channelName`` the user puts in the device tree.

    Class attributes (structured metadata — no magic values in method bodies):
      ``SENSORS``: the 14 degradation-bearing sensor column names
        (drops constant cols 1/5/6/10/16/18/19).
      ``SUPPORTED_SUBSETS``: C-MAPSS subsets with test data available.
    """

    # 14 sensors that carry degradation info.
    SENSORS: list[str] = [
        f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)
    ]
    SUPPORTED_SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

    def __init__(self, data_dir: str | Path, subset: str = "FD001") -> None:
        subset = subset.upper()
        if subset not in self.SUPPORTED_SUBSETS:
            raise ValueError(f"Unsupported C-MAPSS subset: {subset}")
        self.subset = subset
        self._prefix = f"CMAPSS_{subset}_"
        test_path = Path(data_dir) / f"test_{subset}.txt"
        if not test_path.exists():
            raise FileNotFoundError(f"C-MAPSS test file not found: {test_path}")

        df = pd.read_csv(test_path, sep=r"\s+", header=None, names=_CMAPSS_COLS)
        # Per-engine raw sensor arrays (n_engines list of (T_i, 14) arrays).
        self._engines: list[np.ndarray] = []
        self._channel_to_idx: dict[str, int] = {}
        for i, unit_id in enumerate(sorted(df["unit"].unique())):
            mask = df["unit"] == unit_id
            sensors = df.loc[mask, self.SENSORS].to_numpy(dtype=np.float32)
            self._engines.append(sensors)
            self._channel_to_idx[f"{self._prefix}{int(unit_id)}"] = i
        # Per-engine playback pointer (how many cycles have been "observed").
        # Start at 1 so the first window is just the first cycle (padded).
        self._pointers: list[int] = [1] * len(self._engines)
        self._lock = threading.Lock()
        logger.info(
            "CMAPSSDataSource loaded %s: %d engines, channel range %s..%s",
            subset, len(self._engines),
            f"{self._prefix}1", f"{self._prefix}{len(self._engines)}",
        )

    # ── RulDataSource protocol ─────────────────────────────────────────

    def channels(self) -> list[str]:
        return list(self._channel_to_idx.keys())

    def get_window(self, channel: str, n_cycles: int) -> np.ndarray | None:
        idx = self._channel_to_idx.get(channel)
        if idx is None:
            return None
        sensors = self._engines[idx]
        ptr = self._pointers[idx]
        if ptr <= 0:
            return None
        available = sensors[:ptr]  # (ptr, 14) raw
        if len(available) < n_cycles:
            # Front-pad with the earliest row (matches RULPredictor padding).
            pad = np.tile(available[:1], (n_cycles - len(available), 1))
            return np.vstack([pad, available]).astype(np.float32)
        return available[-n_cycles:].astype(np.float32)

    def advance(self) -> None:
        with self._lock:
            for i, sensors in enumerate(self._engines):
                nxt = self._pointers[i] + 1
                # Loop back to the start once an engine is exhausted so the
                # demo never runs dry.  Reset to 1 (not 0) so the next window
                # is immediately non-empty.
                self._pointers[i] = nxt if nxt <= len(sensors) else 1


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RulService:
    """Long-horizon RUL prediction service polled by the front-end.

    Holds a dict of :class:`RULPredictor` instances keyed by model id
    (``"fd001"`` etc.) and a matching dict of :class:`RulDataSource`
    instances.  Each call to :meth:`predict_all` advances every data source
    by one cycle and predicts RUL for every channel whose device-tree
    description carries a ``@rul:<model>`` tag.

    Multi-source routing: each tagged channel is routed to the data source
    whose key matches its model id.  This lets fd001 and fd003 coexist —
    each subset has its own CMAPSSDataSource reading its own test_FD00X.txt.
    """

    def __init__(
        self,
        data_sources: dict[str, RulDataSource],
        predictors: dict[str, RULPredictor],
        config_service: ConfigService,
        window_cycles: int = 30,
        history_len: int = 20,
    ) -> None:
        self._sources = data_sources
        self._predictors = predictors
        self._config = config_service
        self.window_cycles = window_cycles
        self.history_len = history_len
        # Per-channel rolling RUL history (most recent first appended).
        self._history: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    # ── Device-tree tag scan ───────────────────────────────────────────

    def channels_with_rul(self) -> dict[str, str]:
        """Return ``{channelName: model_id}`` for sensors tagged ``@rul:``.

        Reads the current device tree from ConfigService on each call so
        background edits (e.g. via SimpleUI admin) are picked up live.
        """
        cfg = self._config.load()
        tree = cfg.get("device_tree", [])
        out: dict[str, str] = {}

        def walk(nodes: list) -> None:
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                if n.get("type") == "sensor":
                    model_id = parse_rul_tag(n.get("description"))
                    if model_id and model_id in self._predictors:
                        ch = n.get("channelName") or n.get("name")
                        if ch:
                            out[ch] = model_id
                children = n.get("children")
                if children:
                    walk(children)

        walk(tree)
        return out

    def _source_for(self, model_id: str) -> RulDataSource | None:
        """Return the data source registered under ``model_id``, or None."""
        return self._sources.get(model_id)

    # ── Prediction ─────────────────────────────────────────────────────

    def predict_all(self) -> list[dict]:
        """Advance one cycle and predict RUL for every tagged channel.

        Returns a list of result dicts (one per channel that currently has
        enough data).  Each dict carries a short ``history`` list for the
        front-end trend sparkline.
        """
        with self._lock:
            for src in self._sources.values():
                src.advance()
            tagged = self.channels_with_rul()
            results: list[dict] = []
            for channel, model_id in tagged.items():
                source = self._source_for(model_id)
                if source is None:
                    continue
                window = source.get_window(channel, self.window_cycles)
                if window is None:
                    continue
                predictor = self._predictors[model_id]
                rul = predictor.predict_rul(window, raw=True)
                hist = self._history.setdefault(channel, [])
                hist.append(rul)
                if len(hist) > self.history_len:
                    del hist[: len(hist) - self.history_len]
                results.append({
                    "channel": channel,
                    "rul": round(float(rul), 1),
                    "max_rul": int(predictor.max_rul),
                    "unit": "cycles",
                    "model": model_id,
                    "source": f"C-MAPSS {model_id.upper()}（基准演示）",
                    "history": list(hist),
                })
            return results

    def predict(self, channel: str) -> dict | None:
        """Predict RUL for a single channel without advancing the pointer.

        Used for ad-hoc /api/rul?channel=xxx queries.  Does not append to
        history (predict_all owns the trend stream).
        """
        tagged = self.channels_with_rul()
        model_id = tagged.get(channel)
        if model_id is None:
            return None
        source = self._source_for(model_id)
        if source is None:
            return None
        window = source.get_window(channel, self.window_cycles)
        if window is None:
            return None
        rul = self._predictors[model_id].predict_rul(window, raw=True)
        return {
            "channel": channel,
            "rul": round(float(rul), 1),
            "max_rul": int(self._predictors[model_id].max_rul),
            "unit": "cycles",
            "model": model_id,
            "source": f"C-MAPSS {model_id.upper()}（基准演示）",
            "history": list(self._history.get(channel, [])),
        }

    # ── Agent-friendly introspection (shared by CLI and /api/rul/status) ──

    def status(self) -> dict:
        """Return a structured snapshot of the RUL service state.

        Exposes the loaded models, their data sources, and the currently
        tagged channels.  Used by ``manage.py rul status`` and the API so
        agents and operators can inspect what is enabled without parsing
        logs.  Pure read — no side effects.
        """
        return {
            "enabled_models": sorted(self._predictors.keys()),
            "data_sources": {
                mid: sorted(src.channels())
                for mid, src in self._sources.items()
            },
            "tagged_channels": self.channels_with_rul(),
            "window_cycles": self.window_cycles,
            "history_len": self.history_len,
        }
