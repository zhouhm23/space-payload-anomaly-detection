"""Database layer: real-time in-memory RingBuffer + historical/intermediate/
result partition placeholders.

This iteration only implements the **real-time** RingBuffer (migrated from
``server.py``'s ``ring_buffers`` dict).  Historical, intermediate and result
partitions are reserved as future extension points per the paper's PHM
database layer spec — they are intentionally left as TODO hooks, not fake
implementations.
"""

from .ring_buffer import RingBuffer, ChannelStore

__all__ = ["RingBuffer", "ChannelStore"]
