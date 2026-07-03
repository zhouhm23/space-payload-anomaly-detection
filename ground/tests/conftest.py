"""Pytest configuration — adds ground/ and space/ to sys.path for imports."""

import os
import sys

_GROUND = os.path.join(os.path.dirname(__file__), "..")
if _GROUND not in sys.path:
    sys.path.insert(0, _GROUND)

# ground tests also need space modules (anomaly_detection is shared)
_SPACE = os.path.join(os.path.dirname(__file__), "..", "..", "space")
if _SPACE not in sys.path:
    sys.path.insert(0, _SPACE)
