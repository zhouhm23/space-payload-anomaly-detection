"""Root URL configuration for PHM Django project."""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('phm_site.urls')),
]
