"""AppConfig for phm_site — starts background threads on ready()."""
import os
import sys

from django.apps import AppConfig


class PhmSiteConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'phm_site'
    verbose_name = 'PHM数据管理'

    def ready(self):
        # Start background threads eagerly when running the server. The
        # services_bridge.get_container() also lazy-inits as a fallback, so
        # if ready() runs in a process that doesn't serve requests, the
        # serving process will init on first request.
        is_runserver = 'runserver' in sys.argv
        is_main = os.environ.get('RUN_MAIN') == 'true'
        is_noreload = '--noreload' in sys.argv
        if not is_runserver:
            return
        if is_main or is_noreload:
            from . import services_bridge
            services_bridge.start()
