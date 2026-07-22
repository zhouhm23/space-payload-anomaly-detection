"""Django settings for PHM ground system (Django + DRF + SimpleUI).

工程结构（v1.1 重写）：
- 前台监控大屏：Vue3 SPA，build 后产物落到 django_phm/static/phm_site/dist/
- 后台管理：SimpleUI 主体 + 自定义 Django 模板（extends admin/base_site.html）
- API：Django REST Framework（/api/v2/* 新规范）+ 保留旧视图过渡（/api/*）

业务逻辑层（src/ground/phm/）零改动，通过 services_bridge 桥接 Container。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# src/ground/django_phm/settings.py → src/ground/
BASE_DIR = Path(__file__).resolve().parent.parent  # src/ground
SRC_DIR = BASE_DIR.parent  # src/

# 把 src/ground/ 加入 sys.path，使 `import phm` / `import comm` 可用
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── .env 加载（从 server.py 迁移） ──────────────────────────────────────────
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

# ── HuggingFace 离线缓存（必须在 from_pretrained 之前设置） ────────────────
# 历史教训（Day18）：不设 HF_HUB_OFFLINE 会联网确认 revision，触发 meta-tensor
# 损坏后续模型构造。必须强制离线 + 本地快照。
_HF_CACHE = SRC_DIR / ".hf_cache"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ── Django 核心 ─────────────────────────────────────────────────────────────
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
    # 第三方
    'rest_framework',
    'django_filters',
    'corsheaders',
    # 本项目
    'phm_site',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # 必须在 CommonMiddleware 之前
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

# ── 本地化（SimpleUI 后台简体中文） ────────────────────────────────────────
LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = False

# ── 数据库（共用 phm.db） ───────────────────────────────────────────────────
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': str(BASE_DIR / 'data' / 'phm.db'),
    }
}

# ── 静态文件 ────────────────────────────────────────────────────────────────
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'django_phm' / 'static',  # 自定义页静态资源
]

# 生产部署：collectstatic 收集到此目录
STATIC_ROOT = BASE_DIR / 'django_phm' / 'staticfiles'

# Vue3 前台大屏 build 产物（生产环境直接 serve）
# 开发环境通过 vite dev server (:5173) 代理，不走这里
FRONTEND_DIST = BASE_DIR / 'django_phm' / 'static' / 'phm_site' / 'dist'
if FRONTEND_DIST.exists():
    STATICFILES_DIRS.append(FRONTEND_DIST)

# ── DRF 配置 ────────────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',  # 后台同源
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',  # 前台大屏匿名访问，细粒度在视图层
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
    'DATETIME_FORMAT': None,  # 返回原始 ISO 字符串
    'DEFAULT_TIME_ZONE': 'UTC',
}

# ── CORS（Vue3 dev server :5173 → Django :8501） ────────────────────────────
CORS_ORIGIN_ALLOW_ALL = True  # 开发期允许所有源（生产环境应配白名单）
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_ALL_METHODS = True
CORS_ALLOW_ALL_HEADERS = True

# ── SimpleUI 配置 ───────────────────────────────────────────────────────────
SIMPLEUI_HOME_INFO = False
SIMPLEUI_ANALYSIS = False
# 登录后台默认跳转仪表盘（v1.1 新增）
SIMPLEUI_INDEX = '/admin/phm_site/dashboard/'
# 品牌名：登录页 / 首页大标题（原默认 "Django administration"，用户反馈应改为本系统简称）
SIMPLEUI_HOME_TITLE = '天地PHM 管理后台'
# 自定义菜单（对齐需求书 §后台 9 项）：首页/用户管理/审计日志走 SimpleUI 默认，
# 仪表盘/告警与预警/回收站/设备树/系统设置/模型管理为本系统自定义页。
# system_keep=True 让 SimpleUI 自动列出 django.contrib.auth (User/Group →「认证和授权」)
# 与 django.contrib.admin (LogEntry →「管理」/审计日志) 的默认菜单组。
# menu_display 是白名单 + 排序：精确控制只显示需求书要求的 4 个组，
# 隐藏 SimpleUI 默认会列出的 PHM数据管理（业务表的 ModelAdmin 列表，开发用不上面向用户）。
# 「权限说明」是用户管理页上的按钮（需求书原文："用户管理...加个说明按钮，
# 打开显示权限说明面板"），URL 保留但不挂菜单——按钮嵌入待用户管理页改造时完成。
SIMPLEUI_CONFIG = {
    # 品牌名（左侧菜单头部主标题 + 副标题 + 图标）。fas fa-satellite-dish
    # 契合「天地协同」（卫星↔地面站）语义。
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
                {'name': '回收站', 'icon': 'fas fa-trash', 'url': '/admin/phm_site/recycle/'},
            ],
        },
        {
            'name': '配置管理',
            'icon': 'fas fa-cog',
            'models': [
                {'name': '设备树', 'icon': 'fas fa-sitemap', 'url': '/admin/phm_site/device-tree/'},
                {'name': '系统设置', 'icon': 'fas fa-sliders-h', 'url': '/admin/phm_site/settings/'},
                {'name': '模型管理', 'icon': 'fas fa-cubes', 'url': '/admin/phm_site/models/'},
            ],
        },
    ],
}

# ── PHM 运行时配置 ──────────────────────────────────────────────────────────
SPACE_HOST = os.environ.get("SPACE_HOST", "127.0.0.1")
SPACE_PORT = int(os.environ.get("SPACE_PORT", "9876"))
PHM_CONFIG_PATH = str(BASE_DIR / "device_config.json")

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── 日志 ────────────────────────────────────────────────────────────────────
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
