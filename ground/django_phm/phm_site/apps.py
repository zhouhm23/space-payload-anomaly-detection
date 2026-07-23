"""AppConfig for phm_site — starts the background threads on boot (auto-poll + eval + auto-diagnosis)."""
import os
import sys

from django.apps import AppConfig


class PhmSiteConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'phm_site'
    verbose_name = 'PHM数据管理'

    def ready(self):
        # Only start the background threads under runserver.
        # services_bridge.get_container() also has a lazy-init fallback, so even
        # if ready() runs in a non-serving process (e.g. migrate) it will not block.
        is_runserver = 'runserver' in sys.argv
        is_main = os.environ.get('RUN_MAIN') == 'true'
        is_noreload = '--noreload' in sys.argv
        if not is_runserver:
            return
        if is_main or is_noreload:
            from . import services_bridge
            services_bridge.start()
