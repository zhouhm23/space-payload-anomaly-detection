"""Model registry — single source of truth for model identifiers.

Centralises the HuggingFace hub ids, local weight paths, and architectural
constants that were previously hard-coded as module-level ``DEFAULT_MODEL``
strings scattered across ``ttm.py`` / ``tspulse.py`` / ``rul_model.py``.

Why a registry (not just config.py constants)?
  * Hub ids are tightly coupled to the loader class (each model needs a
    specific ``from_pretrained`` subclass).  Co-locating the id with a
    reference to the loader keeps them in sync.
  * Future model swaps (e.g. a fine-tuned TSPulse checkpoint) only need an
    entry here — no edits to the loader modules.
  * Agent-friendly: ``manage.py models`` can enumerate every model the
    system knows about without importing heavy torch modules.

This is a *metadata* registry — it does not import the model classes
themselves (those pull in tsfm_public / torch and are expensive).  Each
loader module reads its own entry lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ModelEntry", "MODEL_REGISTRY", "get_model_entry"]


@dataclass(frozen=True)
class ModelEntry:
    """Metadata for one loadable model.

    Attributes:
        key: short identifier used in config / logs (e.g. ``"tspulse"``).
        kind: role — ``"detector"`` / ``"forecaster"`` / ``"rul"``.
        hub_id: HuggingFace model id, or empty for local-weight models.
        context_length: input window length the model was trained on.
        prediction_length: output horizon (0 for detectors / RUL).
        notes: free-text provenance / version note.
        deploy: deployment target — ``"ground"`` (local inference on the
            ground segment) / ``"space"`` (space segment, OTA push reserved).
            Space-segment models are intended to be pushed to onboard compute
            nodes via OTA in the future; for now this is a metadata-only flag
            and triggers no transport logic.
    """
    key: str
    kind: str
    hub_id: str
    context_length: int
    prediction_length: int
    notes: str = ""
    deploy: str = "ground"


# The three foundation / trained models the system currently ships with.
# When a model is swapped (e.g. a fine-tuned TSPulse r2), add a new entry
# here and point the loader at it — old entries stay for reproducibility.
MODEL_REGISTRY: dict[str, ModelEntry] = {
    "tspulse": ModelEntry(
        key="tspulse",
        kind="detector",
        hub_id="ibm-granite/granite-timeseries-tspulse-r1",
        context_length=512,
        prediction_length=0,
        notes="Zero-shot anomaly detection (TSB-UAD benchmark). R1 release.",
    ),
    "ttm_r3": ModelEntry(
        key="ttm_r3",
        kind="forecaster",
        hub_id="ibm-research/ttm-r3",
        context_length=512,
        prediction_length=96,
        notes="Zero-shot forecasting (512→96). Revision 512-96-dec-512-r3, "
              "decomposed-prediction variant.",
    ),
    "rul": ModelEntry(
        key="rul",
        kind="rul",
        hub_id="",  # local LSTM+Attention weights, not on HF hub
        context_length=30,
        prediction_length=1,
        notes="Supervised RUL on C-MAPSS (FD001 RMSE=14.88). "
              "Weights under models/rul/, scaler JSON alongside.",
    ),
}


def get_model_entry(key: str) -> ModelEntry | None:
    """Return the registry entry for ``key``, or None if unknown."""
    return MODEL_REGISTRY.get(key)
