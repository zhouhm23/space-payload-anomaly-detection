"""Root pytest configuration — shared across all test directories.

Sets HuggingFace cache environment variables so tests reuse the offline
model cache under ``src/.hf_cache`` instead of re-downloading on every run.

Background: ``server.py`` and ``space/main.py`` set these at module import
time, but pytest imports the detector modules directly (without going
through those entry points), so the cache dir defaulted to
``~/.cache/huggingface`` — a different directory from the runtime cache.
This conftest closes that gap.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Point HF cache at the same directory used by the runtime (server.py /
# main.py both set this).  Must be set *before* any transformers/tsfm_public
# import, so it lives at module top level.
os.environ.setdefault("HF_HOME", os.path.join(_HERE, ".hf_cache"))
# Offline mode: once the model is cached locally, never hit the network.
# Avoids slow/failed SSL HEAD requests to huggingface.co on every run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
