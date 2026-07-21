"""WSGI config for django_phm project."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_phm.settings')

application = get_wsgi_application()
