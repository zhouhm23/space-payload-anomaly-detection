"""phm_site URL routes."""
from django.urls import path

from . import views

urlpatterns = [
    # monitor page (real-time dashboard, migrated from frontend/dashboard.html)
    path('monitor/', views.monitor_view),
    # monitor embedded in SimpleUI admin (iframe wrapper, staff login required)
    path('monitor-embed/', views.monitor_embed_view),
    # poll / forecast / config / reset / health / sensors
    path('api/poll', views.poll_view),
    path('api/forecast', views.forecast_view),
    path('api/config', views.config_view),       # GET + POST merged
    path('api/reset', views.reset_view),
    path('api/health', views.health_view),
    path('api/sensors', views.sensors_view),
    # alerts (4)
    path('api/alerts', views.alerts_view),
    path('api/alerts/history', views.alerts_history_view),
    path('api/alerts/verdict', views.alert_verdict_view),
    path('api/alerts/<int:alert_id>', views.patch_alert_view),
    # warnings (3)
    path('api/warnings', views.warnings_view),
    path('api/predict-scores', views.predict_scores_view),
    path('api/warnings/<int:warning_id>/verdict', views.warning_verdict_view),
    # history (merged GET+DELETE) / detection (merged GET+DELETE) / db-stats
    path('api/history', views.history_view),       # GET + DELETE merged
    path('api/detection', views.detection_view),   # GET + DELETE merged
    path('api/db-stats', views.db_stats_view),
    # window + export
    path('api/window', views.window_view),
    path('api/export', views.export_view),
    # diagnosis (4)
    path('api/diagnosis', views.diagnosis_view),
    path('api/diagnosis/done', views.diagnosis_done_view),
    path('api/diagnosis/auto', views.diagnosis_auto_view),
    path('api/diagnosis/auto/status', views.diagnosis_auto_status_view),
    # RUL degradation prediction
    path('api/rul', views.rul_view),
]
