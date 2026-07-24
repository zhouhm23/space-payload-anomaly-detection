"""Django settings for PHM ground system (Django + DRF + SimpleUI).

Project layout (v1.1 rewrite):
- Front-end monitor: Vue3 SPA; build output lands in django_phm/static/phm_site/dist/
- Admin: SimpleUI body + custom Django templates (extends admin/base_site.html)
- API: Django REST Framework (/api/v2/* new spec) + legacy views kept for transition (/api/*)

The business-logic layer (src/ground/phm/) is untouched and is bridged into
Django via services_bridge.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# src/ground/django_phm/settings.py → src/ground/
BASE_DIR = Path(__file__).resolve().parent.parent  # src/ground
SRC_DIR = BASE_DIR.parent  # src/

# Add src/ground/ to sys.path so `import phm` / `import comm` work
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── .env loading (migrated from server.py) ──────────────────────────────────
def _load_dotenv() -> None:
    env_path = SRC_DIR / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    except Exception:
        pass


_load_dotenv()

# ── HuggingFace offline cache (must be set before any from_pretrained) ──────
# Historical lesson (Day18): without HF_HUB_OFFLINE the loader pings the hub to
# confirm the revision, which triggers meta-tensor corruption in subsequent
# model construction. Offline mode + a local snapshot are mandatory.
_HF_CACHE = SRC_DIR / ".hf_cache"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ── Django core ─────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-insecure-key-change-in-prod')
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'simpleui',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # third-party
    'rest_framework',
    'django_filters',
    'corsheaders',
    # this project
    'phm_site',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # must precede CommonMiddleware
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'django_phm.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'django_phm' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'phm_site.context_processors.theme',
            ],
        },
    },
]

WSGI_APPLICATION = 'django_phm.wsgi.application'
ASGI_APPLICATION = 'django_phm.asgi.application'

# ── Localisation (SimpleUI admin in Simplified Chinese) ─────────────────────
LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = False

# ── Database (shared phm.db) ────────────────────────────────────────────────
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': str(BASE_DIR / 'data' / 'phm.db'),
    }
}

# ── Static files ────────────────────────────────────────────────────────────
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'django_phm' / 'static',  # custom-page static assets
]

# Production: collectstatic gathers files here
STATIC_ROOT = BASE_DIR / 'django_phm' / 'staticfiles'

# Vue3 front-end monitor build output (served directly in production).
# In development the vite dev server (:5173) proxies it, so this path is unused.
FRONTEND_DIST = BASE_DIR / 'django_phm' / 'static' / 'phm_site' / 'dist'
if FRONTEND_DIST.exists():
    STATICFILES_DIRS.append(FRONTEND_DIST)

# ── DRF config ──────────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',  # same-origin with the admin
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',  # the monitor is anonymous; fine-grained control lives in the views
    ],
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    'DATETIME_FORMAT': None,  # return the raw ISO string
    'DEFAULT_TIME_ZONE': 'UTC',
}

# ── CORS (Vue3 dev server :5173 → Django :8501) ─────────────────────────────
CORS_ORIGIN_ALLOW_ALL = True  # allow all origins in dev (production should use a whitelist)
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_ALL_METHODS = True
CORS_ALLOW_ALL_HEADERS = True

# ── SimpleUI config ─────────────────────────────────────────────────────────
SIMPLEUI_HOME_INFO = False
SIMPLEUI_ANALYSIS = False
# Target URL for the top-right fa-home button's window.open().
# Points to /monitor/ (the front-end monitor) — reuses SimpleUI's built-in home
# button as the "admin → front-end" jump, so dashboard.html needs no custom
# button. The dashboard is still reachable from the left menu.
SIMPLEUI_INDEX = '/monitor/'
# Brand name: login page / home heading (Django's default "Django administration";
# changed to this system's short name per user feedback)
SIMPLEUI_HOME_TITLE = '天地PHM 管理后台'
# Custom menu (aligned with spec admin section — 9 items): home/user-management/audit-log
# go through SimpleUI defaults; dashboard/alert-management/recycle/device-tree/
# system-settings/model-management are this system's custom pages.
# system_keep=True lets SimpleUI auto-list django.contrib.auth (User/Group →
# "认证和授权") and django.contrib.admin (LogEntry → "管理" / audit log) default
# menu groups.
# menu_display is a whitelist + ordering: it shows exactly the 4 groups the spec
# requires and hides the "PHM数据管理" group SimpleUI would list by default
# (the ModelAdmin list of business tables — a dev aid, not user-facing).
# "权限说明" is a button on the user-management page (spec verbatim: "user
# management ... add a button that opens a permissions panel"); the URL is kept
# but not wired into the menu — the embedded button lands when the
# user-management page is reworked.
SIMPLEUI_CONFIG = {
    # Brand name (left-menu heading + subtitle + icon). fas fa-satellite-dish
    # matches the "space-ground synergy" (satellite ↔ ground station) semantics.
    'title': '天地PHM',
    'subtitle': '空间载荷健康管理平台',
    'icon': 'fas fa-satellite-dish',
    'system_keep': True,
    'menu_display': ['运营管理', '配置管理', '认证和授权', '管理'],
    'menus': [
        {
            'name': '运营管理',
            'icon': 'fas fa-chart-line',
            'models': [
                {'name': '仪表盘', 'icon': 'fas fa-tachometer-alt', 'url': '/admin/phm_site/dashboard/'},
                {'name': '告警与预警', 'icon': 'fas fa-bell', 'url': '/admin/phm_site/alert/'},
                {'name': '遥测数据', 'icon': 'fas fa-database', 'url': '/admin/phm_site/telemetry/'},
                # v1.2: recycle bin splits into per-data-type sub-menus (three-level).
                # SimpleUI renders a nested ``models`` list as a third level under
                # the parent entry. Both children share one ``recycle_view``,
                # dispatched by the ``?type=`` query string.
                {'name': '回收站', 'icon': 'fas fa-trash', 'models': [
                    {'name': '告警回收站', 'icon': 'fas fa-bell-slash',
                     'url': '/admin/phm_site/recycle/?type=alert'},
                    {'name': '遥测回收站', 'icon': 'fas fa-database',
                     'url': '/admin/phm_site/recycle/?type=telemetry'},
                ]},
            ],
        },
        {
            'name': '配置管理',
            'icon': 'fas fa-cog',
            'models': [
                {'name': '设备树', 'icon': 'fas fa-sitemap', 'url': '/admin/phm_site/device-tree/'},
                {'name': '系统设置', 'icon': 'fas fa-sliders-h', 'url': '/admin/phm_site/settings/'},
                {'name': '算法库', 'icon': 'fas fa-cubes', 'url': '/admin/phm_site/library/'},
            ],
        },
    ],
}

# ── PHM runtime config ──────────────────────────────────────────────────────
SPACE_HOST = os.environ.get("SPACE_HOST", "127.0.0.1")
SPACE_PORT = int(os.environ.get("SPACE_PORT", "9876"))
PHM_CONFIG_PATH = str(BASE_DIR / "device_config.json")

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── Logging ─────────────────────────────────────────────────────────────────
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{asctime}] {levelname} {name} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'phm_site': {'level': 'INFO', 'handlers': ['console'], 'propagate': False},
        'phm': {'level': 'WARNING', 'handlers': ['console'], 'propagate': False},
    },
    'root': {'level': 'WARNING', 'handlers': ['console']},
}
