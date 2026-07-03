"""Pytest configuration — adds src/ to sys.path for e2e tests."""

import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
