"""API routes — thin FastAPI routers that delegate to services.

Each route module is independent so they can be tested in isolation.  The
shared dependency container (``deps.py``) holds the singleton services.
"""

from . import deps
from .routes_poll import router as poll_router
from .routes_forecast import router as forecast_router
from .routes_config import router as config_router
from .routes_reset import router as reset_router
from .routes_health import router as health_router
from .routes_alerts import router as alerts_router
from .routes_warnings import router as warnings_router
from .routes_sensors import router as sensors_router
from .routes_history import router as history_router
from .routes_window import router as window_router
from .routes_export import router as export_router
from .routes_diagnosis import router as diagnosis_router

__all__ = [
    "deps",
    "poll_router",
    "forecast_router",
    "config_router",
    "reset_router",
    "health_router",
    "alerts_router",
    "warnings_router",
    "sensors_router",
    "history_router",
    "window_router",
    "export_router",
    "diagnosis_router",
]
