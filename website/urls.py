from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload'),
    path('process/<str:filename>/', views.process_image, name='process'),
    path("overlay/", views.overlay_page, name="overlay_page"),
    path("options/", views.options_page, name="options_page"),
    path("store-points/", views.store_points, name="store_points"),

]
