"""AppConfig for phm_site — 启动时拉起后台线程（auto-poll + eval + auto-diagnosis）."""
import os
import sys

from django.apps import AppConfig


class PhmSiteConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'phm_site'
    verbose_name = 'PHM数据管理'

    def ready(self):
        # 仅在 runserver 时启动后台线程。
        # services_bridge.get_container() 也有 lazy-init 兜底，
        # 因此即便 ready() 跑在非服务进程（如 migrate），也不会阻塞。
        is_runserver = 'runserver' in sys.argv
        is_main = os.environ.get('RUN_MAIN') == 'true'
        is_noreload = '--noreload' in sys.argv
        if not is_runserver:
            return
        if is_main or is_noreload:
            from . import services_bridge
            services_bridge.start()
