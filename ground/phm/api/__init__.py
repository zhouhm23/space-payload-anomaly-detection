"""API dependency container (shared by Django services_bridge).

The FastAPI route modules that used to live here were removed when the
project migrated to Django (``django_phm``).  The HTTP surface is now
served by Django views (``phm_site/views.py``).  This package retains
only the dependency container (``deps``), which both the Django bridge
and the unit tests import to construct the service singletons
(RingBuffer, SQLiteStore, WarningService, etc.).
"""

from . import deps

__all__ = ["deps"]
