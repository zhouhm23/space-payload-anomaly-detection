"""ASGI config for django_phm project."""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_phm.settings')

application = get_asgi_application()
