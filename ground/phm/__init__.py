"""PHM four-layer architecture (ground segment).

Layers (paper-aligned):
  - database/    : in-memory RingBuffer (real-time), historical/intermediate/
                   result partitions reserved as extension points
  - dataops/     : space-side preprocessing reuse + feature-extraction plugin
  - algorithm/   : TSPulse (detection) + TTM-R3 (forecast), unified plugin base
  - model/       : model registry (placeholder, not implemented this iteration)

The legacy flat modules (``forecasting.py``, ``anomaly_detection.py``,
ring buffers in ``server.py``) have been migrated here.  Public API aliases
are kept at the ``ground`` package root so existing import sites keep working
during the transition.
"""

__all__ = ["config"]
