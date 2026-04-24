from django.urls import include, path

urlpatterns = [
    path("", include("unified_api.urls")),
]
