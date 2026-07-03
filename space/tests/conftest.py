"""Pytest configuration — adds space/ to sys.path for local imports."""

import os
import sys

_SPACE = os.path.join(os.path.dirname(__file__), "..")
if _SPACE not in sys.path:
    sys.path.insert(0, _SPACE)
